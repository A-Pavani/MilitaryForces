#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void loading_bar() {
  int i;
  int bar_length = 50; // Length of the loading bar
  printf("\nLoading: [");
  fflush(stdout); // Flush the output buffer to ensure consistent display

  for (i = 0; i < bar_length; i++) {
    usleep(70000);  // Delay of 60 milliseconds for each step
    printf("█");    // Use a solid block character for the bar
    fflush(stdout); // Ensure the output is displayed immediately
  }

  printf("] Done!\n\n");
}

void display_data_types() {
  printf("\n=================================================\n");
  printf("               DATA TYPES MENU\n");
  printf("=================================================\n");
  printf("1. Text Data\n");
  printf("2. Image Data\n");
  printf("3. Tabular Data\n");
  printf("=================================================\n");
}

void display_goals() {
  printf("\n=================================================\n");
  printf("                 GOALS MENU\n");
  printf("=================================================\n");
  printf("1. Classification\n");
  printf("2. Regression\n");
  printf("=================================================\n");
}

void display_format_suggestions(const char *data_type) {
  printf("\n=================================================\n");
  printf("           FORMAT AND LIBRARY SUGGESTIONS\n");
  printf("=================================================\n");

  if (strcmp(data_type, "Text") == 0) {
    printf("Recommended Formats: CSV, TXT, JSON\n");
    printf("Library Suggestions:\n");
    printf("  1. **pandas**: Use for reading CSV, TXT, and JSON files into "
           "DataFrames.\n");
    printf("     - Example: `df = pd.read_csv('yourfile.csv')`\n");
    printf("     - Example: `df = pd.read_json('yourfile.json')`\n");
    printf("  2. **zipfile**: If your files are compressed in a ZIP archive, "
           "use the `zipfile` library to extract them.\n");
    printf("     - Example:\n");
    printf("       ```python\n");
    printf("       import zipfile\n");
    printf("       with zipfile.ZipFile('yourfile.zip', 'r') as zip_ref:\n");
    printf("           zip_ref.extractall('your_extraction_path')\n");
    printf("       ```\n");

  } else if (strcmp(data_type, "Image") == 0) {
    printf("Recommended Formats: JPEG, PNG, TIFF\n");
    printf("Library Suggestions:\n");
    printf("  1. **tensorflow**: Use `tf.io.read_file` for loading image "
           "files.\n");
    printf("     - Example: `img = tf.io.read_file('yourimage.jpg')`\n");
    printf("  2. **opencv**: Use for reading and processing image files.\n");
    printf("     - Example: `img = cv2.imread('yourimage.png')`\n");
    printf("  3. **PIL** (Python Imaging Library): Use for advanced image "
           "manipulation and format conversions.\n");
    printf("     - Example: `from PIL import Image`\n");
    printf("     - Example: `img = Image.open('yourimage.tiff')`\n");
    printf("  4. **zipfile**: If your images are stored in a ZIP archive, "
           "extract them using `zipfile`.\n");
    printf("     - Example:\n");
    printf("       ```python\n");
    printf("       import zipfile\n");
    printf("       with zipfile.ZipFile('images.zip', 'r') as zip_ref:\n");
    printf("           zip_ref.extractall('image_extraction_path')\n");
    printf("       ```\n");

  } else if (strcmp(data_type, "Tabular") == 0) {
    printf("Recommended Formats: CSV, Excel, JSON\n");
    printf("Library Suggestions:\n");
    printf("  1. **pandas**: Use for loading and processing CSV, Excel, and "
           "JSON files.\n");
    printf("     - Example: `df = pd.read_csv('yourfile.csv')`\n");
    printf("     - Example: `df = pd.read_excel('yourfile.xlsx')`\n");
    printf("     - Example: `df = pd.read_json('yourfile.json')`\n");
    printf(
        "  2. **openpyxl**: Use for more advanced Excel file manipulations.\n");
    printf("     - Example: `from openpyxl import load_workbook`\n");
    printf("     - Example: `wb = load_workbook('yourfile.xlsx')`\n");
    printf("  3. **zipfile**: If your tabular data is compressed in a ZIP "
           "archive, use the `zipfile` library to extract it.\n");
    printf("     - Example:\n");
    printf("       ```python\n");
    printf("       import zipfile\n");
    printf("       with zipfile.ZipFile('data.zip', 'r') as zip_ref:\n");
    printf("           zip_ref.extractall('data_extraction_path')\n");
    printf("       ```\n");
  }
  printf("=================================================\n");
}

void display_preprocessing_steps(const char *data_type) {
  printf("\n=================================================\n");
  printf("            DATA CLEANING & PREPROCESSING\n");
  printf("=================================================\n");

  if (strcmp(data_type, "Text") == 0) {
    printf("• **Text Cleaning**:\n");
    printf("  - Convert all text to lowercase to ensure uniformity.\n");
    printf("  - Remove punctuation, special characters, and digits.\n");
    printf("  - Remove stopwords (common words like 'the', 'and', etc.) using "
           "libraries like 'nltk'.\n");
    printf("  - Tokenization: Split text into individual words or tokens.\n");
    printf("  - Stemming/Lemmatization: Reduce words to their base or root "
           "form.\n");
    printf("\n• **Handling Imbalanced Classes**:\n");
    printf("  - Oversampling: Duplicate samples from the minority class.\n");
    printf("  - Undersampling: Remove samples from the majority class.\n");
    printf("  - Use class weights: Assign higher weights to the minority class "
           "during model training.\n");

  } else if (strcmp(data_type, "Image") == 0) {
    printf("• **Image Resizing**:\n");
    printf("  - Ensure all images are resized to a consistent dimension "
           "suitable for your model.\n");
    printf("  - Common sizes include 224x224, 256x256, etc., depending on the "
           "architecture.\n");
    printf("\n• **Normalization**:\n");
    printf("  - Normalize pixel values from the range 0-255 to a range of 0-1 "
           "or -1 to 1.\n");
    printf("  - This helps the model converge faster during training.\n");
    printf("\n• **Image Augmentation**:\n");
    printf("  - Apply random transformations to images to create variations "
           "and prevent overfitting.\n");
    printf("  - Techniques include rotations, flips, zooms, shifts, brightness "
           "adjustments, etc.\n");
    printf("  - Libraries like TensorFlow, Keras, or OpenCV can be used for "
           "augmentation.\n");
    printf("\n• **Class Balancing**:\n");
    printf("  - Use augmentation techniques to generate more images for the "
           "minority class.\n");
    printf("  - Alternatively, sample less from the majority class.\n");

  } else if (strcmp(data_type, "Tabular") == 0) {
    printf("• **Handling Missing Values**:\n \033[0m ");
    printf("  - Imputation: Fill missing values using the mean, median, or "
           "mode of the column.\n");
    printf("  - Removal: Drop rows or columns with a significant number of "
           "missing values.\n");
    printf("\n• **Feature Scaling**:\n");
    printf("  - Normalize or standardize numerical features to ensure they are "
           "on the same scale.\n");
    printf("  - Methods include Min-Max Scaling (scaling between 0-1) or "
           "Z-score normalization (mean=0, std=1).\n");
    printf("\n• **Encoding Categorical Variables**:\n");
    printf("  - Convert categorical variables into numerical format using "
           "techniques like One-Hot Encoding.\n");
    printf("  - Label Encoding can be used for ordinal data where the order "
           "matters.\n");
    printf("\n• **Class Balancing**:\n");
    printf("  - Use oversampling or undersampling techniques to balance class "
           "distributions.\n");
    printf("  - SMOTE (Synthetic Minority Over-sampling Technique) can be "
           "applied to generate synthetic samples.\n");
    printf("\n• **Outlier Detection**:\n");
    printf("  - Detect and handle outliers using methods like z-score, IQR "
           "(Interquartile Range), or visualizations.\n");
    printf("  - Depending on the context, outliers can be capped, transformed, "
           "or removed.\n");
  }
  printf("=================================================\n");
}

void display_algorithms(bool is_classification, const char *data_type) {
  printf("\n=================================================\n");
  printf("            RECOMMENDED ALGORITHMS\n");
  printf("=================================================\n");

  if (is_classification) {
    if (strcmp(data_type, "Text") == 0 || strcmp(data_type, "Tabular") == 0) {
      printf("\n1. [Logistic Regression]\n");
      printf("---------------------------------------------\n");
      printf("   Description: Used for binary classification problems where\n");
      printf("                the output variable is categorical.\n");
      printf("   Applications: Spam detection, disease prediction, etc.\n");
      printf("   Note: Logistic Regression is often the go-to choice for\n");
      printf(
          "         binary classification tasks, especially with text data.\n");
      printf("         For more complex relationships, consider SVM or Random "
             "Forest.\n");
      printf("=============================================\n");

      printf("\n2. [Random Forest Classifier]\n");
      printf("---------------------------------------------\n");
      printf("   Description: An ensemble learning method that builds multiple "
             "decision\n");
      printf("                trees and merges them for more accurate and "
             "stable predictions.\n");
      printf(
          "   Applications: Image classification, feature selection, etc.\n");
      printf("   Note: Random Forest is preferred when you need high accuracy "
             "and a robust model.\n");
      printf("         It's particularly useful when the dataset has many "
             "features.\n");
      printf("=============================================\n");

      printf("\n3. [Support Vector Machines (SVM)]\n");
      printf("---------------------------------------------\n");
      printf("   Description: A classification algorithm that finds the "
             "hyperplane\n");
      printf("                best separating different classes in the feature "
             "space.\n");
      printf("   Applications: Text classification, image recognition, etc.\n");
      printf("   Note: SVM is highly effective for high-dimensional spaces.\n");
      printf("         It's a top choice for text classification tasks,\n");
      printf("         but can be computationally expensive for large "
             "datasets.\n");
      printf("=============================================\n");

      printf("\n4. [Naive Bayes]\n");
      printf("---------------------------------------------\n");
      printf("   Description: A probabilistic classifier based on applying "
             "Bayes' theorem\n");
      printf("                with strong (naive) independence assumptions "
             "between features.\n");
      printf("   Applications: Spam filtering, sentiment analysis, etc.\n");
      printf("   Note: Naive Bayes is often used for text classification due "
             "to its\n");
      printf("         simplicity and speed. It's particularly effective when "
             "features\n");
      printf("         are independent.\n");
      printf("=============================================\n");

    } else if (strcmp(data_type, "Image") == 0) {
      printf("\n1. [Convolutional Neural Networks (CNNs)]\n");
      printf("---------------------------------------------\n");
      printf("   Description: A class of deep neural networks, most commonly "
             "applied\n");
      printf("                to analyzing visual imagery.\n");
      printf("   Applications: Image classification, object detection, etc.\n");
      printf("   Note: CNNs are the standard for image-related tasks.\n");
      printf(
          "         For simpler problems with smaller datasets, a standard\n");
      printf("         neural network may suffice.\n");
      printf("=============================================\n");

      printf("\n2. [Transfer Learning with Pre-trained Models]\n");
      printf("---------------------------------------------\n");
      printf("   Description: Utilizes pre-trained models like VGG16, ResNet, "
             "or Inception,\n");
      printf("                which have been trained on large datasets like "
             "ImageNet.\n");
      printf("   Applications: Fine-tuning these models for specific image "
             "classification tasks.\n");
      printf("   Note: Transfer learning is highly effective when you have a "
             "small dataset.\n");
      printf("         It allows you to leverage the learned features of "
             "complex models.\n");
      printf("=============================================\n");

      printf("\n3. [K-Nearest Neighbors (KNN) for Image Classification]\n");
      printf("---------------------------------------------\n");
      printf("   Description: A simple, instance-based learning algorithm used "
             "for\n");
      printf("                classifying images based on their similarity to "
             "other images.\n");
      printf("   Applications: Basic image recognition tasks where "
             "computational cost is not a concern.\n");
      printf("   Note: KNN is easy to implement and interpret, but is less "
             "efficient with large datasets.\n");
      printf("         It can be a good choice for prototyping or educational "
             "purposes.\n");
      printf("=============================================\n");

      printf("\n4. [Support Vector Machines (SVM) with Image Features]\n");
      printf("---------------------------------------------\n");
      printf("   Description: SVM can be applied to features extracted from "
             "images, such as using\n");
      printf("                Histogram of Oriented Gradients (HOG) or other "
             "feature descriptors.\n");
      printf("   Applications: Image classification tasks, especially when "
             "using traditional methods\n");
      printf("                that don't rely on deep learning.\n");
      printf("   Note: SVM can be effective for image classification when "
             "combined with powerful\n");
      printf("         feature extraction techniques. It's often used when you "
             "want to avoid\n");
      printf("         the complexity of deep learning.\n");
      printf("=============================================\n");
    }
  } else { // Regression
    printf("\n1. [Linear Regression]\n");
    printf("---------------------------------------------\n");
    printf("   Description: A simple algorithm used for predicting\n");
    printf("                a continuous target variable based on\n");
    printf("                the linear relationship between input features.\n");
    printf(
        "   Applications: Predicting house prices, sales forecasting, etc.\n");
    printf("   Note: Developers generally opt for Linear Regression when\n");
    printf("         the relationship between variables is linear and the\n");
    printf("         output is continuous.\n");
    printf("=============================================\n");

    printf("\n2. [Decision Trees]\n");
    printf("---------------------------------------------\n");
    printf("   Description: A non-linear algorithm that splits data into "
           "branches\n");
    printf("                to predict a target variable.\n");
    printf("   Applications: Customer segmentation, loan approval prediction, "
           "etc.\n");
    printf("   Note: Decision Trees are favored for their interpretability.\n");
    printf("         However, they can overfit on small datasets.\n");
    printf("         For improved accuracy, consider Random Forest.\n");
    printf("=============================================\n");

    printf("\n3. [K-Nearest Neighbors (KNN)]\n");
    printf("---------------------------------------------\n");
    printf("   Description: A simple, instance-based learning algorithm used "
           "for\n");
    printf("                both classification and regression.\n");
    printf("   Applications: Recommender systems, image recognition, etc.\n");
    printf("   Note: KNN is easy to implement and works well with smaller "
           "datasets.\n");
    printf(
        "         It's not ideal for large datasets due to its high memory\n");
    printf("         and computation requirements.\n");
    printf("=============================================\n");

    printf("\n4. [Principal Component Analysis (PCA)]\n");
    printf("---------------------------------------------\n");
    printf("   Description: A dimensionality reduction technique that "
           "transforms\n");
    printf("                high-dimensional data into a lower-dimensional "
           "form while\n");
    printf("                retaining as much variability as possible.\n");
    printf("   Applications: Data visualization, noise reduction, etc.\n");
    printf(
        "   Note: PCA is essential when dealing with high-dimensional data.\n");
    printf("         It reduces the number of features, making the dataset "
           "easier\n");
    printf("         to manage, but can result in loss of information.\n");
    printf("=============================================\n");
  }
}

int read_integer(int lower, int upper) {
  int value;
  int len = 0;
  int invalid = 0;
  int valid = 0;
  int decimal_counter = 0;
  char *buffer;
  int buffer_size = 40;
  buffer =
      (char *)malloc(buffer_size * sizeof(char)); // Dynamic memory allocation

  enum Validation { NO_ERROR, INVALID_INPUT, INVALID_RANGE };
  const char *Validation_messages[] = {
      "",
      "\nThe provided input is incorrect because it is not an integer. Please "
      "ensure that the input is a valid number without any decimals or "
      "alphabets.",
      "\nThe provided input is incorrect because it is not in the appropriate "
      "range. Please ensure that the input is within the allowed range."};
  int validation_status = NO_ERROR;

  while (!valid) {
    fgets(buffer, buffer_size, stdin);
    buffer[strcspn(buffer, "\r\n")] = 0; // Remove newline character

    int i = 0;
    if (buffer[i] == '-') {
      i = 1; // Allow negative sign at the beginning
    }

    for (; i < strlen(buffer); i++) {
      if (buffer[i] == '.') {
        decimal_counter++;
      }
      if ((!isdigit(buffer[i])) && (buffer[i] != '.')) {
        invalid = 1;
        break;
      }
    }

    if (decimal_counter != 0) {
      invalid = 1;
      decimal_counter = 0;
    } else if ((sscanf(buffer, "%d%n", &value, &len) != 1) ||
               (len != strlen(buffer))) {
      invalid = 1;
    } else if ((invalid == 0) && (value < lower || value > upper)) {
      printf("%-s", Validation_messages[INVALID_RANGE]);
      continue;
    } else {
      valid = 1;
      invalid = 0;
      decimal_counter = 0;
    }

    if (invalid) {
      printf("%s", Validation_messages[INVALID_INPUT]);
      invalid = 0;
      decimal_counter = 0;
    } else {
      return value;
    }
  }

  free(buffer); // Free the allocated memory
  return value;
}

int select_data_type() {
  printf("\n Enter your choice: ");
  int data_type = read_integer(1, 3);

  switch (data_type) {
  case 1:
    printf("You selected Text Data.\n");
    break;
  case 2:
    printf("You selected Image Data.\n");
    break;
  case 3:
    printf("You selected Tabular Data.\n");
    break;
  default:
    printf("Invalid selection. Please try again.\n");
    select_data_type(); // Recursively call until a valid option is selected
    break;
  }
  return data_type;
}

int select_goal() {
  printf("\n Enter your choice: ");
  int goal = read_integer(1, 2);

  switch (goal) {
  case 1:
    printf("You selected Classification.\n");
    break;
  case 2:
    printf("You selected Regression.\n");
    break;
  default:
    printf("Invalid selection. Please try again.\n");
    select_goal(); // Recursively call until a valid option is selected
    break;
  }
  return goal;
}

int main() {
  int data_type_choice, goal_choice;
  char data_type[20];

  // Step 1: Display Data Types Menu
  display_data_types();
  data_type_choice = select_data_type();

  // Set the data type based on user's choice
  switch (data_type_choice) {
  case 1:
    strcpy(data_type, "Text");
    break;
  case 2:
    strcpy(data_type, "Image");
    break;
  case 3:
    strcpy(data_type, "Tabular");
    break;
  default:
    printf("Invalid choice! Exiting...\n");
    return 1;
  }

  // Step 2: Display Goals Menu
  display_goals();
  goal_choice = select_goal();

  bool is_classification = (goal_choice == 1);
  loading_bar();

  // Step 3: Display Format and Library Suggestions
  display_format_suggestions(data_type);

  loading_bar();
  // Step 4: Display Preprocessing and Cleaning Steps
  display_preprocessing_steps(data_type);
  loading_bar();

  // Step 5: Display Relevant Algorithms
  display_algorithms(is_classification, data_type);

  return 0;
}