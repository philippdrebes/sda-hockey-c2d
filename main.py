import os
import time
import pickle
from ocean_lib.example_config import get_config_dict
from ocean_lib.ocean.ocean import Ocean
from eth_account import Account
from decimal import Decimal
import argparse

from c2d import dispatcher

config = get_config_dict("mumbai")
ocean = Ocean(config)

# Create OCEAN object. ocean_lib knows where OCEAN is on all remote networks
OCEAN = ocean.OCEAN_token

SAVE_FILE_PATH = "vars.pkl"


def main():
    parser = argparse.ArgumentParser(description="CLI tool for handling ice hockey C2D")

    # Add arguments to the parser
    parser.add_argument("-a", "--algo_did", type=str, help="DID for the algorithm")
    parser.add_argument("-d", "--data_did", type=str, help="DID for the data")
    parser.add_argument("-f", "--file", action="store_true",
                        help="Boolean flag to determine if algorithm and data should be loaded from disk")
    parser.add_argument("-p", "--publish", action="store_true",
                        help="Boolean flag to determine if data should be published")

    args = parser.parse_args()

    print(f"Algo DID: {args.algo_did}")
    print(f"Data DID: {args.data_did}")
    print(f"Publish: {args.publish}")

    if args.publish and (args.algo_did or args.data_did):
        raise Exception("Cannot publish and load from disk at the same time")
    if not args.publish and not args.file:
        raise Exception("Must either publish or load from disk")
    if not args.publish and args.file and not data_exists_on_disk(SAVE_FILE_PATH):
        raise Exception("Cannot load from disk if no data exists on disk")
    if args.publish and not (os.getenv('REMOTE_TEST_PRIVATE_KEY1') and os.getenv('REMOTE_TEST_PRIVATE_KEY2')):
        raise Exception("Cannot publish if no private keys are set")

    (data_wallet, algo_wallet) = setup_wallets()

    if data_exists_on_disk(SAVE_FILE_PATH):
        (data_did, algo_did) = load_from_disk(SAVE_FILE_PATH)
    else:
        data_did = args.data_did
        algo_did = args.algo_did

    if args.publish:
        print("Publishing...")
        (data_ddo, algo_ddo) = publish(data_wallet, algo_wallet)
        save_to_disk((data_ddo.did, algo_ddo.did), SAVE_FILE_PATH)
    else:
        print("Loading...")
        data_ddo = ocean.assets.resolve(data_did)
        algo_ddo = ocean.assets.resolve(algo_did)

    run(data_ddo, algo_ddo, algo_wallet)


def setup_wallets():
    # Create wallets
    data_wallet_private_key = os.getenv('REMOTE_TEST_PRIVATE_KEY1')
    data_wallet = Account.from_key(private_key=data_wallet_private_key)
    assert ocean.wallet_balance(data_wallet) > 0, "data_wallet needs MATIC"
    assert OCEAN.balanceOf(data_wallet) > 0, "data_wallet needs OCEAN"

    algo_wallet_private_key = os.getenv('REMOTE_TEST_PRIVATE_KEY2')
    algo_wallet = Account.from_key(private_key=algo_wallet_private_key)
    assert ocean.wallet_balance(algo_wallet) > 0, "algo_wallet needs MATIC"
    assert OCEAN.balanceOf(algo_wallet) > 0, "algo_wallet needs OCEAN"

    return data_wallet, algo_wallet


def publish(data_wallet, algo_wallet):
    # Publish data
    data_url = "https://raw.githubusercontent.com/philippdrebes/sda-hockey-c2d/main/data/data.csv"
    (data_data_nft, data_datatoken, data_ddo) = dispatcher.publish_data(data_wallet, data_url)

    # Publish algorithm
    (algo_data_nft, algo_datatoken, algo_ddo) = dispatcher.publish_algo(data_wallet)

    data_ddo = dispatcher.allow_algo_to_data(data_ddo, algo_ddo, data_wallet)
    dispatcher.acquire_tokens(data_datatoken, algo_datatoken, data_wallet, algo_wallet)

    return data_ddo, algo_ddo


def run(data_ddo, algo_ddo, algo_wallet):
    (job_id, compute_service) = dispatcher.start_compute_job(data_ddo.did, algo_ddo.did, algo_wallet)

    # Wait until job is done
    succeeded = False
    for _ in range(0, 200):
        status = ocean.compute.status(data_ddo, compute_service, job_id, algo_wallet)
        if status.get("dateFinished") and Decimal(status["dateFinished"]) > 0:
            succeeded = True
            break
        time.sleep(5)

    # Retrieve algorithm output and log files
    output = ocean.compute.compute_job_result_logs(data_ddo, compute_service, job_id, algo_wallet)
    print(f"Output: {output}")


def save_to_disk(variables, path):
    with open(path, 'wb') as f:
        pickle.dump(variables, f)


def load_from_disk(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def data_exists_on_disk(path):
    return os.path.exists(path)


if __name__ == "__main__":
    main()
