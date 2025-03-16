# Define el módulo de procesamiento de los datos para entrenar el modelo de R2GenGPT. 
# Carga y limpia los textos de los informes médicos
# Procesa las imágenes con el Swin Transformer
# Organiza los datos en conjuntos de train/val/test
import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor


class FieldParser:
    # Procesamiento de las imágenes y textos 
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)


    def _parse_image(self, img):
        """
        Procesamiento de las imágenes
        Convierte la imagen en tensores (pixel_values) para ser usados por un modelo de visión (ViT o Swin Transformer)
        """
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        """
        Limpieza del texto
        Elimina ruido y caracteres innecesarios y aplica diferentes reglas según el dataset
        """
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            # Quita numeraciones y signos de puntuación innecesarios
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            # Quita numeraciones, puntuación, saltos de línea y caracteres especiales
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report


    def parse(self, features):
        """
        Conversión de los datos
        Inputs:
            - features [Dict]: diccionario con el id, texto y ruta de la imagen para cada informe 
        Returns: 
            - Dict: diccionario con el {id, input_text, image} de los informes
        """
        to_return = {'id': features['id']}
        report = features.get("report", "")
        # Limpia el reporte
        report = self.clean_report(report)
        to_return['input_text'] = report
        # chest x-ray images
        images = []
        for image_path in features['image_path']:
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                # Carga y transforma las imágenes en tensores
                image = self._parse_image(array)
                images.append(image)
        # Devuelve el diccionario listo para el modelo
        to_return["image"] = images
        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    # Carga y organiza los datos
    def __init__(self, args, split='train'):
        self.args = args
        # Lee los datos desde los archivos json de anotaciones
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    # Crea los conjuntos de train/val/test
    # Los datos pueden ser usados con un DataLoader de PyTorch
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


