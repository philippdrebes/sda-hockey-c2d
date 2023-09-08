from ocean_lib.example_config import get_config_dict
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.ocean.util import to_wei

# Create Ocean instance
config = get_config_dict("mumbai")
ocean = Ocean(config)

# Create OCEAN object. ocean_lib knows where OCEAN is on all remote networks
OCEAN = ocean.OCEAN_token


def publish_data(from_wallet, data_url):
    (data_nft, datatoken, ddo) = ocean.assets.create_url_asset("Ice Hockey Data", data_url, {"from": from_wallet},
                                                               with_compute=True, wait_for_aqua=True)

    print("Just published asset (data):")
    print(f"  data_nft: symbol={data_nft.symbol()}, address={data_nft.address}")
    print(f"  datatoken: symbol={datatoken.symbol()}, address={datatoken.address}")
    print(f"  did={ddo.did}")

    return data_nft, datatoken, ddo


def publish_algo(from_wallet):
    (data_nft, datatoken, ddo) = ocean.assets.create_algo_asset("Ice Hockey Algo",
                                                                url="https://raw.githubusercontent.com/philippdrebes/sda-hockey-c2d/main/algos/example_algo.py",
                                                                tx_dict={"from": from_wallet},
                                                                image="ghcr.io/philippdrebes/sda-hockey-c2d",
                                                                checksum="sha256:d91a2fe9524c679a920b39f4f2070180bf4a8321438ca2b8353b2ca64106fd19",
                                                                tag="main",
                                                                wait_for_aqua=True)

    print("Just published asset (algo):")
    print(f"  data_nft: symbol={data_nft.symbol()}, address={data_nft.address}")
    print(f"  datatoken: symbol={datatoken.symbol()}, address={datatoken.address}")
    print(f"  did={ddo.did}")

    return data_nft, datatoken, ddo


def allow_algo_to_data(data_ddo, algo_ddo, from_wallet):
    compute_service = data_ddo.services[1]
    compute_service.add_publisher_trusted_algorithm(algo_ddo)
    data_ddo = ocean.assets.update(data_ddo, {"from": from_wallet})
    return data_ddo


def acquire_tokens(data_datatoken, algo_datatoken, from_wallet, to_wallet):
    # Alice mints DATA datatokens and ALGO datatokens to Bob.
    # Alternatively, Bob might have bought these in a market.
    data_datatoken.mint(to_wallet, to_wei(5), {"from": from_wallet})
    algo_datatoken.mint(to_wallet, to_wei(5), {"from": from_wallet})


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
    return job_id, compute_service
