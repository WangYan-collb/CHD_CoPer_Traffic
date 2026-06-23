from src.cli.run_suite import build_parser


def test_run_suite_parser_accepts_suite_and_smoke():
    parser = build_parser()
    args = parser.parse_args(["--suite", "configs/rl/comparison_suite.yaml", "--smoke"])
    assert args.suite == "configs/rl/comparison_suite.yaml"
    assert args.smoke is True
