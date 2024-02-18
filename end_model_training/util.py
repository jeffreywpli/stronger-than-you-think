import torch
import numpy as np
import inflection
import sys
import contextlib
import scipy


def signal_handler(sig, frame):
    print('Terminating experiments')
    sys.exit(0)


def pipeline_loader(pipeline, *args, **kwargs):
    seed = kwargs["seed"]
    device = kwargs["device"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # automatically assigns devices based on the GPU count
    kwargs["device"] = device + ":" + str(((seed - 1) % torch.cuda.device_count()))

    return pipeline(*args, **kwargs)


def get_filename(data, pipeline, label_model, end_model, backbone, hard_label):
    em_name = backbone + "_" + end_model if end_model == "Cosine" else end_model

    la = "hard" if hard_label else "soft"
    # TODO Add whether the sampling is stratafied

    return '-'.join(map(inflection.underscore, [data.replace('-', "_"), pipeline.replace('-', "_"), label_model, em_name, la]))


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def has_saturated(sample_mean, sample_std, oracle_mean, oracle_std, num_samples, significance=0.05):

    # statistically tests whether the mean of the sample distribution is less than that of the oracle
    test = scipy.stats.ttest_ind_from_stats(sample_mean, sample_std, num_samples, oracle_mean, oracle_std, num_samples,
                                            alternative="less",  equal_var=False)

    return test.pvalue > significance