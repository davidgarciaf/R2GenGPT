from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from dataset.data_helper import create_datasets



class DataModule(LightningDataModule):
    # Subclase de LightningDataModule que gestiona la carga de datos en PyTorch Lightning. 
    # Facilita el manejo de datasets y DataLoaders en entornos de entrenamiento distribuido

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args

    def prepare_data(self):
        # Está vacío pero serviría para descargar los datos (si fuera necesario), tokenizarlos antes de la carga, preprocesar imágenes y cargarlas en cache.
        """
        Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.

        download

        tokenize

        etc…
        :return:
        """

    def setup(self, stage: str):
        # Carga los datasets de train/val/test y los almacena en self.dataset
        """
        There are also data operations you might want to perform on every GPU. Use setup to do things like:

        count number of classes

        build vocabulary

        perform train/val/test splits

        apply transforms (defined explicitly in your datamodule or assigned in init)

        etc…
        :param stage:
        :return:
        """
        train_dataset, dev_dataset, test_dataset = create_datasets(self.args)
        self.dataset = {
            "train": train_dataset, "validation": dev_dataset, "test": test_dataset
        }


    def train_dataloader(self):
        # Crea un DataLoader para el conjunto de entrenamiento
        # Parámetros:
        #     batch_size: tamaño del lote
        #     drop_last: descarta el último lote si es incompleto
        #     pin_memory: optimiza el uso de memoria en GPU
        #     num_workers: usa múltiples núcleos para acelerar la carga 
        #     prefetch_factor: reduce latencia precargando datos
        """
        Use this method to generate the train dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=True,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader


    def val_dataloader(self):
        """
        Use this method to generate the val dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(self.dataset["validation"], batch_size=self.args.val_batch_size, drop_last=False, pin_memory=True,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader


    def test_dataloader(self):
        loader = DataLoader(self.dataset["test"], batch_size=self.args.test_batch_size, drop_last=False, pin_memory=False,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        return loader
