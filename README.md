# Hyppopipe Framework

> Modern framework for ML-pipelines creation.

**Developer**: Lanin George (TG: [@LaninGM](https://t.me/LaninGM))

The user (ML engineer) sets his own pipeline, defines the steps, the tasks to be solved (transformation, localization, classification, segmentation), sets up training configurations, exports training results, runs the pipeline for the image, gets the results for each step.

The project is being developed by the project [group](cs.hse.ru/dse/ismed ) "Information systems for medical applications — ISMed".

## Documentation

Read interactive docs [here](https://hse-cs.github.io/hyppopipe/en/)

Documentation is available at folder `/docs` in English and Russian.

## Install

Create and activate .venv:

```shell
python -m venv .venv
source .venv/bin/activate
# .\.venv\Scripts\activate # For windows
```

Install hyppopipe package from [TestPyPi](https://test.pypi.org/project/hyppopipe/):

```shell
pip install -i https://test.pypi.org/simple/ hyppopipe
```

## Usage

### Dataset reading

```python
from hyppopipe.data import YAMLDataset, ImageFolderDataset, PairedImageMaskFolderDataset, split_random_fractions

image_folder = ImageFolderDataset(root="/datasets/Medical-imaging-dataset")

masks_dataset = PairedImageMaskFolderDataset(
    "/datasets/NailSegmentation/",
    image_folder="images",
    mask_folder="labels",
)

yolo_dataset = YAMLDataset("/datasets/BrainTumor/dataset.yaml", strict=False)
```

### Dataset loaders

```python
data_split = dataset.as_split_data(fractions=(0.7, 0.15, 0.15))
# or
data_split = split_random_fractions(dataset, (0.8, 0.2))
```

### Pipeline

```python
nails_pipe = Pipeline(
    steps={
        "sharpen": Step(ImageTransformer().sharpen(2.0)),
        "segment": Step(ImageSegmentator(kind="semantic")),
    }
)
```

### Training

```python
from hyppopipe.train import Trainer, ModelCandidate, TrainingConfig

result = nails_pipe.train(
    data=nails_split,
    step_config={
        "segment": Trainer(
            model_candidates=[
                ModelCandidate(
                    deeplabv3_resnet50, weights=[DeepLabV3_ResNet50_Weights.DEFAULT, ]
                ),
            ],
            # data=nails_split, # Separate split also supported
            config=TrainingConfig(
                epochs=20,
                device="mps",
                batch_size=8,
            ),
        )
    }
)
```

### Export

```python
result.export_artifacts(Path("artifacts/nails_seg"), nails_pipe)
```

### Prediction

```python
nail_image = Image.from_path("datasets/nail.jpg")
pred_res = nails_pipe.predict(nail_image, bundle_path=Path("artifacts/nails_seg"))
pred_res.outputs["segment"].show()
```

# Фреймворк Hyppopipe

> Фреймворк для построения ML-пайплайнов в медицинских системах

**Разработчик**: Георгий Ланин (TG: [@LaninGM](https://t.me/LaninGM))

Пользователь (ML-инженер) задаёт свой пайплайн, определяет шаги, решаемые задачи (трансформация, локализация, классификация, сегментация), настраивает конфигурации обучения, экспортирует результаты обучения, запускает пайплайн для изображения, получает результаты для каждого шага.

Проект развивается проектной [группой](cs.hse.ru/dse/ismed) «Информационные системы для медицинских приложений — ИСМед».

## Установка

Создаём и активируем виртуальное окружение .venv

```shell
python -m venv .venv
source .venv/bin/activate
# .\.venv\Scripts\activate # For windows
```

Устанавливаем зависимости проекта из индекса [TestPyPi](https://test.pypi.org/project/hyppopipe/).

```shell
pip install -i https://test.pypi.org/simple/ hyppopipe
```

## Документация

Интерактивная документация [здесь](https://hse-cs.github.io/hyppopipe/ru/)

Документация доступна в паке `/docs` на английском и русском языках.

## Использование

### Чтение датасетов

```python
from hyppopipe.data import YAMLDataset, ImageFolderDataset, PairedImageMaskFolderDataset, split_random_fractions

image_folder = ImageFolderDataset(root="/datasets/Medical-imaging-dataset")

masks_dataset = PairedImageMaskFolderDataset(
    "/datasets/NailSegmentation/",
    image_folder="images",
    mask_folder="labels",
)

yolo_dataset = YAMLDataset("/datasets/BrainTumor/dataset.yaml", strict=False)
```

### Загрузчки датасетов для обучения

```python
dataset.as_split_data(fractions=(0.7, 0.15, 0.15))
# or
split_random_fractions(dataset, (0.8, 0.2))
```

### Описание пайплайнов

```python
nails_pipe = Pipeline(
    steps={
        "sharpen": Step(ImageTransformer().sharpen(2.0)),
        "segment": Step(ImageSegmentator(kind="semantic")),
    }
)
```

### Обучение

```python
result = nails_pipe.train(
    data=nails_split,
    step_config={
        "segment": Trainer(
            model_candidates=[

                ModelCandidate(
                    deeplabv3_resnet50, weights=[DeepLabV3_ResNet50_Weights.DEFAULT, ]
                ),
            ],
            data=nails_split,
            config=TrainingConfig(
                epochs=20,
                device="mps",
                batch_size=8,
            ),
        )
    }
)
```

### Экспорт результатов обучения

```python
result.export_artifacts(Path("artifacts/nails_seg"), nails_pipe)
```

### Предсказание

```python
nail_image = Image.from_path("datasets/nail.jpg")
pred_res = nails_pipe.predict(nail_image, bundle_path=Path("artifacts/nails_seg"))
pred_res.outputs["segment"].show()
```