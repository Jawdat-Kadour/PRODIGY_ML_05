# Food Recognition and Calorie Estimation from Images

This project combines food recognition with calorie estimation, providing a solution to help users estimate the calorie content of meals based on food images. The system recognizes food items from images using a pre-trained deep learning model (EfficientNetB3), estimates portion sizes, and retrieves nutritional information to compute the total calorie content.

## Introduction

This project aims to automate the process of food recognition and calorie estimation using a deep learning model for image classification and a nutritional database for calorie retrieval. This is particularly useful for diet tracking, fitness apps, and meal planning.

## Project Workflow

1. **Image Input**: Users provide an image of a food item.
2. **Food Recognition**: A pre-trained model (EfficientNetB3) identifies the food item in the image.
3. **Portion Size Estimation**: Users input the estimated portion size (in grams) or the portion size is estimated using additional sensors or tools.
4. **Calorie Estimation**: The calorie content is calculated based on the food item and portion size using a calorie API (such as USDA FoodData Central).

## Features

- Recognizes food items from images using a pre-trained EfficientNetB3 model.
- Estimates portion size and calculates the calorie content of the meal.
- Retrieves calorie information using OpenAI API or the USDA FoodData Central API.
- Supports customizable portion sizes.
  
## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Jawdat-Kadour/food-recognition-calorie-estimation.git
    cd food-recognition-calorie-estimation
    ```

2. **Install dependencies**:
    Use `pip` to install the required packages.
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the Dataset**:
    This project uses the **Food-101 dataset**. Download the dataset from [here](https://www.vision.ee.ethz.ch/datasets_extra/food-101/), then extract it and place the data in the `data/` folder.

4. **Set up the USDA API Key** (if using the USDA FoodData Central API):
    - Go to the [USDA FoodData Central](https://fdc.nal.usda.gov/api-key-signup.html) to get your API key.
    - Add your API key to the `get_calories_usda` function in the script.

## Dataset

The **Food-101 dataset** contains 101 food categories, each with 1,000 images. The dataset is split into training and validation sets, where 750 images are used for training and 250 for validation per class.

- **Training data**: Images are resized to 299x299 for compatibility with the EfficientNetB3 model.
- **Data Augmentation**: Applied during training to prevent overfitting.

## Model Architecture

This project uses a **pre-trained EfficientNetB3** model for food recognition, which is fine-tuned on the Food-101 dataset. The architecture includes:
- EfficientNetB3 as the base model (pre-trained on ImageNet).
- Fully connected layers for classification.
- Global Average Pooling and Batch Normalization layers for regularization.
- Dropout layers to reduce overfitting.

## Usage

1. **Training the Model**:
    To train the model on the dataset, run the following command:
    ```bash
    python train.py
    ```
    The model will be trained on the Food-101 dataset with various data augmentation techniques to enhance generalization.

2. **Calorie Estimation**:
    Once a food item is recognized, the user is prompted to enter the portion size (in grams), and the total calorie count is calculated using the `get_calories` function.
    
    Example:
    ```python
    food_item = "banana"
    portion_size = 150  # grams
    calories = get_calories(food_item)
    total_calories = (portion_size / 100) * float(calories.split()[0])
    print(f"Total calories for {portion_size}g of {food_item}: {total_calories} calories")
    ```

3. **Predicting Food from an Image**:
    After training, you can use the model to predict the food item from an image:
    ```python
    python predict.py --image path/to/your/image.jpg
    ```

## Future Improvements

- **Portion Size Estimation**: Use machine learning models or depth sensors to automatically estimate the portion size from images.
- **Multi-Item Food Recognition**: Enable recognition of multiple food items in a single image.
- **Improved Calorie Retrieval**: Integrate additional APIs for more detailed nutritional information.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenAI API (optional for text-based calorie estimation)
- USDA FoodData Central API (optional for accurate calorie retrieval)

Install the required Python packages with:
```bash
pip install -r requirements.txt
```

#### List of Required Libraries:
- `tensorflow`
- `numpy`
- `Pillow`
- `scikit-learn`
- `requests`
- `openai`
