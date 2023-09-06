# Create Ocean instance
import os
import time
import pickle
from ocean_lib.example_config import get_config_dict
from ocean_lib.ocean.ocean import Ocean
from eth_account import Account
from decimal import Decimal

from c2d import dispatcher

config = get_config_dict("mumbai")
ocean = Ocean(config)

# Create OCEAN object. ocean_lib knows where OCEAN is on all remote networks
OCEAN = ocean.OCEAN_token


def publish_and_run():
    data_wallet_private_key = os.getenv('REMOTE_TEST_PRIVATE_KEY1')
    data_wallet = Account.from_key(private_key=data_wallet_private_key)
    assert ocean.wallet_balance(data_wallet) > 0, "data_wallet needs MATIC"
    assert OCEAN.balanceOf(data_wallet) > 0, "data_wallet needs OCEAN"

    # Publish data NFT, datatoken, and asset for dataset based on url
    data_url = "https://raw.githubusercontent.com/oceanprotocol/c2d-examples/main/iris_and_logisitc_regression/dataset_61_iris.csv"
    (data_data_nft, data_datatoken, data_ddo) = dispatcher.publish_data(data_wallet, data_url)

    algo_wallet_private_key = os.getenv('REMOTE_TEST_PRIVATE_KEY2')
    algo_wallet = Account.from_key(private_key=algo_wallet_private_key)
    assert ocean.wallet_balance(algo_wallet) > 0, "algo_wallet needs MATIC"
    assert OCEAN.balanceOf(algo_wallet) > 0, "algo_wallet needs OCEAN"

    container_metadata = {
        "entrypoint": "python algos/example_algo.py",
        "image": "ghcr.io/philippdrebes/sda-hockey-c2d",
        "tag": "main",
        "checksum": "sha256:25fa8b7d99912576508c570ce96d715a81dbe05a47188594d089fc10c51b14ef",
    }

    (algo_data_nft, algo_datatoken, algo_ddo) = dispatcher.publish_algo(algo_wallet, container_metadata)

    data_ddo = dispatcher.allow_algo_to_data(data_ddo, algo_ddo, data_wallet)
    dispatcher.acquire_tokens(data_datatoken, algo_datatoken, data_wallet, algo_wallet)

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
    output = ocean.compute.compute_job_result_logs(
        data_ddo, compute_service, job_id, algo_wallet
    )[0]

    model = pickle.loads(output)  # the result
    assert len(model) > 0, "unpickle result unsuccessful"


if __name__ == '__main__':
    publish_and_run()
