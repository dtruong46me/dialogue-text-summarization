from model import GeneralModel

# FLAN-T5 MODEL
class FlanT5Model(GeneralModel):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)