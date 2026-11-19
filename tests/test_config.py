from dllama import config


def test_server_config_created_with_token(tmp_path):
    path = tmp_path / "config.toml"
    cfg, created = config.load_server(path)
    assert created and path.exists()
    assert cfg["control"]["token"]
    assert cfg["ollama"]["env"]["OLLAMA_CONTEXT_LENGTH"] == "65536"

    again, created_again = config.load_server(path)
    assert not created_again
    assert again["control"]["token"] == cfg["control"]["token"]


def test_env_table_replaces_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[ollama.env]\nOLLAMA_KEEP_ALIVE = "5m"\n', encoding="utf-8")
    cfg, _ = config.load_server(path)
    assert cfg["ollama"]["env"] == {"OLLAMA_KEEP_ALIVE": "5m"}
    assert cfg["proxy"]["bind"] == "0.0.0.0:11434"


def test_ollama_env_drops_inherited_and_forces_host():
    cfg = {"ollama": {"internal_port": 11435, "env": {"OLLAMA_NUM_PARALLEL": 1}}}
    env = config.ollama_env({"PATH": "x", "OLLAMA_HOST": "0.0.0.0:11434", "OLLAMA_KEEP_ALIVE": "5m"}, cfg)
    assert env == {"PATH": "x", "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_HOST": "127.0.0.1:11435"}


def test_client_config_defaults_to_target_host(tmp_path):
    path = tmp_path / "client.toml"
    cfg = config.load_client(path)
    assert path.exists()
    assert (cfg["host"], cfg["port"], cfg["token"]) == ("169.254.51.51", 11500, "")


def test_local_address_maps_wildcard_to_loopback():
    assert config.local_address("0.0.0.0:11500") == "127.0.0.1"
    assert config.local_address(":11500") == "127.0.0.1"
    assert config.local_address("169.254.51.51:11500") == "169.254.51.51"
