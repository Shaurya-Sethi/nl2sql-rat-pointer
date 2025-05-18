from src.config import NL2SQLConfig

def test_pretraining_config():
    config = NL2SQLConfig.from_yaml('src/config.yaml', 'pretraining')
    print(f'Model max_len: {config.max_len}')
    print(f'Phase-specific max_len: {config.phase_max_len}')
    print(f'Dataset max_len from get_dataset_max_len(): {config.get_dataset_max_len()}')
    assert config.max_len == 2048, "Model max_len should be 2048"
    assert config.phase_max_len == 512, "Pretraining phase_max_len should be 512"
    assert config.get_dataset_max_len() == 512, "Dataset max_len should be 512 for pretraining"
    return True

def test_sft_config():
    config = NL2SQLConfig.from_yaml('src/config.yaml', 'sft')
    print(f'Model max_len: {config.max_len}')
    print(f'Phase-specific max_len: {config.phase_max_len}')
    print(f'Dataset max_len from get_dataset_max_len(): {config.get_dataset_max_len()}')
    assert config.max_len == 2048, "Model max_len should be 2048"
    assert config.phase_max_len is None, "SFT phase_max_len should be None"
    assert config.get_dataset_max_len() == 2048, "Dataset max_len should be 2048 for SFT (model's max_len)"
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