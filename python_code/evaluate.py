import os

from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    trainer = DeepSICTrainer()
    print(trainer)
    trainer.evaluate()
