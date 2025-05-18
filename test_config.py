from src.config import NL2SQLConfig

def test_pretraining_config():
    config = NL2SQLConfig.from_yaml('src/config.yaml', 'pretraining')
    print(f'Model max_len: {config.max_len}')
    print(f'Phase-specific dataset_item_max_len: {config.dataset_phase_max_len}')
    print(f'Dataset max_len from get_dataset_max_len(): {config.get_dataset_max_len()}')
    print(f'SFT PG source max_len: {config.phase_max_len_pg}') # Should be None for pretraining
    assert config.max_len == 2048, "Model max_len should be 2048"
    assert config.dataset_phase_max_len == 512, "Pretraining dataset_phase_max_len should be 512"
    assert config.get_dataset_max_len() == 512, "Dataset max_len should be 512 for pretraining"
    assert config.phase_max_len_pg is None, "Pretraining phase_max_len_pg should be None"
    return True

def test_sft_config():
    config = NL2SQLConfig.from_yaml('src/config.yaml', 'sft')
    print(f'Model max_len: {config.max_len}')
    print(f'Phase-specific dataset_item_max_len: {config.dataset_phase_max_len}')
    print(f'SFT PG source (schema+NL) max_len: {config.phase_max_len_pg}')
    print(f'SFT max_sql_len: {config.max_sql_len}')
    print(f'Dataset max_len from get_dataset_max_len(): {config.get_dataset_max_len()}')
    assert config.max_len == 2048, "Model max_len should be 2048"
    # SFT phase in YAML doesn't have its own 'max_len' for dataset items, so it defaults to model max_len
    assert config.dataset_phase_max_len is None, "SFT dataset_phase_max_len should be None (uses model max_len)"
    assert config.get_dataset_max_len() == 2048, "Dataset max_len should be 2048 for SFT (model's max_len)"
    assert config.phase_max_len_pg == 1664, "SFT phase_max_len_pg should be 1664"
    assert config.max_sql_len == 320, "SFT max_sql_len should be 320"
    return True

if __name__ == "__main__":
    print("Testing pretraining config...")
    pretraining_result = test_pretraining_config()
    
    print("\nTesting SFT config...")
    sft_result = test_sft_config()
    
    if pretraining_result and sft_result:
        print("\nConfig tests PASSED!")
    else:
        print("\nConfig tests FAILED!") 