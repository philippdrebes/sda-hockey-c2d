from ocean_lib.example_config import get_config_dict
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.ocean.util import to_wei
from ocean_lib.structures.file_objects import UrlFile

# Create Ocean instance
config = get_config_dict("mumbai")
ocean = Ocean(config)

# Create OCEAN object. ocean_lib knows where OCEAN is on all remote networks
OCEAN = ocean.OCEAN_token


# Publish data NFT, datatoken, and asset for dataset based on url

def publish_data(from_wallet):
    # ocean.py offers multiple file object types. A simple url file is enough for here
    # Specify metadata, using the iris dataset
    data_date_created = "2019-12-28T10:55:11Z"
    data_metadata = {
        "created": data_date_created,
        "updated": data_date_created,
        "description": "The Iris flower dataset is a multivariate dataset to train classification algorithms",
        "name": "Iris Flower Dataset",
        "type": "dataset",
        "author": "Ocean Protocol & Raven Protocol",
        "license": "MIT",
    }

    # ocean.py offers multiple file types, but a simple url file should be enough for this example
    data_url_file = UrlFile(
        url="https://raw.githubusercontent.com/oceanprotocol/c2d-examples/main/iris_and_logisitc_regression/dataset_61_iris.csv"
    )

    (data_data_nft, data_datatoken, data_ddo) = ocean.assets.create_url_asset(*data_metadata, data_url_file.url,
                                                                              {"from": from_wallet},
                                                                              with_compute=True, wait_for_aqua=True)
    print(f"DATA_data_nft address = '{data_data_nft.address}'")
    print(f"DATA_datatoken address = '{data_datatoken.address}'")

    print(f"DATA_ddo did = '{data_ddo.did}'")


def publish_algo(from_wallet):
    # Specify metadata, using the Logistic Regression algorithm
    algo_date_created = "2020-01-28T10:55:11Z"
    algo_metadata = {
        "created": algo_date_created,
        "updated": algo_date_created,
        "description": "Logistic Regression",
        "name": "Logistic Regression",
        "type": "algorithm",
        "author": "Ocean Protocol & Raven Protocol",
        "license": "MIT",
        "algorithm": {
            "language": "python",
            "format": "docker-image",
            "version": "0.1",
            "container": {
                "entrypoint": "python $ALGO",
                "image": "oceanprotocol/algo_dockers",
                "tag": "python-panda",
                # This image provides all the dependencies of the logistic_regression.py algorithm
                "checksum": "sha256:7fc268f502935d11ff50c54e3776dda76477648d5d83c2e3c4fdab744390ecf2",
            },
        }
    }

    # ocean.py offers multiple file types, but a simple url file should be enough for this example
    from ocean_lib.structures.file_objects import UrlFile
    algo_url_file = UrlFile(
        url="https://raw.githubusercontent.com/oceanprotocol/c2d-examples/main/iris_and_logisitc_regression/logistic_regression.py"
    )

    (algo_data_nft, algo_datatoken, algo_ddo) = ocean.assets.create_algo_asset(*algo_metadata, algo_url_file.url,
                                                                               {"from": from_wallet},
                                                                               wait_for_aqua=True)

    print(f"ALGO_data_nft address = '{algo_data_nft.address}'")
    print(f"ALGO_datatoken address = '{algo_datatoken.address}'")
    print(f"ALGO_ddo did = '{algo_ddo.did}'")


def allow_algo_to_data(DATA_ddo, ALGO_ddo, from_wallet):
    compute_service = DATA_ddo.services[1]
    compute_service.add_publisher_trusted_algorithm(ALGO_ddo)
    data_ddo = ocean.assets.update(DATA_ddo, {"from": from_wallet})


def acquire_tokens(DATA_datatoken, ALGO_datatoken, from_wallet, to_wallet):
    # Alice mints DATA datatokens and ALGO datatokens to Bob.
    # Alternatively, Bob might have bought these in a market.
    DATA_datatoken.mint(to_wallet, to_wei(5), {"from": from_wallet})
    ALGO_datatoken.mint(to_wallet, to_wei(5), {"from": from_wallet})


def start_compute_job(data_did, algo_did, consumer_wallet):
    # Operate on updated and indexed assets
    data_ddo = ocean.assets.resolve(data_did)
    algo_ddo = ocean.assets.resolve(algo_did)

    compute_service = data_ddo.services[1]
    algo_service = algo_ddo.services[0]
    free_c2d_env = ocean.compute.get_free_c2d_environment(compute_service.service_endpoint, data_ddo.chain_id)

    from datetime import datetime, timedelta, timezone
    from ocean_lib.models.compute_input import ComputeInput

    data_compute_input = ComputeInput(data_ddo, compute_service)
    algo_compute_input = ComputeInput(algo_ddo, algo_service)

    # Pay for dataset and algo for 1 day
    datasets, algorithm = ocean.assets.pay_for_compute_service(
        datasets=[data_compute_input],
        algorithm_data=algo_compute_input,
        consume_market_order_fee_address=consumer_wallet.address,
        tx_dict={"from": consumer_wallet},
        compute_environment=free_c2d_env["id"],
        valid_until=int((datetime.now(timezone.utc) + timedelta(days=1)).timestamp()),
        consumer_address=free_c2d_env["consumerAddress"],
    )
    assert datasets, "pay for dataset unsuccessful"
    assert algorithm, "pay for algorithm unsuccessful"

    # Start compute job
    job_id = ocean.compute.start(
        consumer_wallet=consumer_wallet,
        dataset=datasets[0],
        compute_environment=free_c2d_env["id"],
        algorithm=algorithm,
    )
    print(f"Started compute job with id: {job_id}")
