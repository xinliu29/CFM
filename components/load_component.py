from . import readers
from . import evaluators
from . import extractors


def load_component(compo_name, model_name, config):
    if compo_name == 'extractor':
        component = load_extractor(model_name, config)
    elif compo_name == 'reader':
        component = load_reader(model_name, config)
    elif compo_name == 'evaluator':
        component = load_evaluator(model_name, config)
    else:
        raise NotImplementedError
    return component


def load_extractor(model_name, config):
    if model_name == 'root':
        extractor = extractors.ExtractSIFT(config)
    elif model_name == 'sp':
        extractor = extractors.ExtractSuperpoint(config)
    else:
        raise NotImplementedError
    return extractor


def load_reader(model_name='standard', config=None):
    if model_name == 'standard':
        reader = readers.standard_reader(config)
    else:
        raise NotImplementedError
    return reader


def load_evaluator(model_name='AUC', config=None):
    if model_name == 'AUC':
        evaluator = evaluators.auc_eval(config)
    elif model_name == 'FM':
        evaluator = evaluators.FMbench_eval(config)
    else:
        raise NotImplementedError
    return evaluator
