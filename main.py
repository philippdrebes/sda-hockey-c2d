# Create Ocean instance
import os
from ocean_lib.example_config import get_config_dict
from ocean_lib.ocean.ocean import Ocean
from eth_account import Account

config = get_config_dict("mumbai")
ocean = Ocean(config)

# Create OCEAN object. ocean_lib knows where OCEAN is on all remote networks
OCEAN = ocean.OCEAN_token

if __name__ == '__main__':
    alice_private_key = os.getenv('REMOTE_TEST_PRIVATE_KEY1')
    alice = Account.from_key(private_key=alice_private_key)
    assert ocean.wallet_balance(alice) > 0, "Alice needs MATIC"
    assert OCEAN.balanceOf(alice) > 0, "Alice needs OCEAN"
