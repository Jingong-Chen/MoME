"""Runtime patch helpers to register RegDBCross without editing existing modules."""


def patch_regdb_cross_factories():
    import importlib

    from datasets.regdb_cross import RegDBCross
    from datasets.sampler_regdb_cross import RandomIdentitySamplerCross
    md = importlib.import_module("datasets.make_dataloader")
    mdm = importlib.import_module("datasets.make_dataloader_mome")
    mdmia = importlib.import_module("datasets.make_dataloader_mia")

    md.__factory["regdb_cross"] = RegDBCross
    mdm.__factory["regdb_cross"] = RegDBCross
    mdmia.__factory["regdb_cross"] = RegDBCross

    # Use balanced cross-modality identity sampling for RegDB cross training.
    md.RandomIdentitySampler = RandomIdentitySamplerCross
    mdm.RandomIdentitySampler = RandomIdentitySamplerCross


def patch_local_factory_dict(factory_dict):
    from datasets.regdb_cross import RegDBCross

    if isinstance(factory_dict, dict):
        factory_dict["regdb_cross"] = RegDBCross


def set_regdb_cross_env(trial=None, test_mode=None):
    import os

    if trial is not None:
        os.environ["REGDB_CROSS_TRIAL"] = str(int(trial))
    if test_mode is not None:
        test_mode = str(test_mode).lower()
        if test_mode not in ("v2t", "t2v"):
            raise ValueError("test_mode must be v2t or t2v")
        os.environ["REGDB_CROSS_TEST_MODE"] = test_mode
