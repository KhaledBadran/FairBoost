import enum

class Preproc_name(str, enum.Enum):
    OptimPreproc = "OptimPreproc"
    LFR = 'LFR'
    DisparateImpactRemover = 'DisparateImpactRemover'
    Reweighing = 'Reweighing'


class Dataset_name(str, enum.Enum):
    GERMAN = 'german'
    ADULT = 'adult'
    COMPASS = 'compas'
