from src.cli.train import build_parser as build_train_parser
from src.cli.meta_train import build_parser as build_meta_train_parser


def test_train_cli_parser_accepts_smoke_flag():
    parser = build_train_parser()
    args = parser.parse_args(["--config", "configs/rl/trans_beta_ppo.yaml", "--smoke"])
    assert args.smoke is True


def test_meta_train_cli_parser_accepts_smoke_flag():
    parser = build_meta_train_parser()
    args = parser.parse_args(["--config", "configs/meta_rl/maml_trans_beta_ppo.yaml", "--smoke"])
    assert args.smoke is True
