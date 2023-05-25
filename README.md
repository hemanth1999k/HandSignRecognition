# Hand Gesture Recognition System using Deep Learning
This project aims to implement a hand gesture recognition system using deep learning techniques. Hand gesture recognition poses several challenges, such as the need to recognize shapes and extract both temporal and spatial information from continuous image sequences. Unlike single images, the hand gestures in this system are represented in video format.

The project proposes two alternative lightweight models as solutions to address these challenges. The first model is a recurrent model inspired by Neural Circuit Policies (NCP), which provides an alternative to the commonly used Long Short-Term Memory (LSTM) model. The second model is the DivAtten XnL model, based on self-attention mechanisms similar to the transformer model.

Both models utilize convolutional neural networks (CNNs) to extract features from input images. The NCP model incorporates recurrence, which is crucial for capturing temporal aspects in the data. By incorporating recurrence, the stability of the DivAtten model in live prediction is significantly improved.

Compared to the popular and computationally intensive model like ResNet50, the proposed lightweight models offer faster processing times, making them suitable for live recognition applications. The DivAtten XnL model introduces attention scores to make the model more selective in identifying features specific to a particular gesture.

Features
Hand gesture recognition system implemented using deep learning techniques
Video-based representation of hand gestures
Two lightweight alternative models: NCP-inspired recurrent model and DivAtten XnL model
Convolutional neural networks for feature extraction
Improved stability in live prediction with the addition of recurrence in the DivAtten model
Faster processing times compared to models like ResNet50

# Results
The performance of the proposed models was evaluated on a benchmark dataset, achieving competitive accuracy rates compared to existing hand gesture recognition systems. The lightweight models demonstrated their efficiency in live recognition scenarios, providing faster processing times without compromising accuracy.

# Future Work
Possible areas for future improvement and expansion of this project include:

Exploration of other lightweight models or architectures for hand gesture recognition
Investigation of different techniques for temporal feature extraction
Integration of real-time hand tracking for improved gesture localization
Development of a user-friendly interface for easy interaction with the system
Deployment of the system on embedded devices for gesture recognition applications in resource-constrained environments
Contributing
Contributions to this project are welcome. If you have any suggestions, bug reports, or feature requests, please submit an issue or a pull request. Let's collaborate to enhance the hand gesture recognition system together.
# Acknowledgments
We would like to express our sincere gratitude to the following contributors who have played a significant role in the development of this project:

[Hemanth Dhanasekaran](https://github.com/hemanth1999k)

[Nanthak Kumar](https://github.com/nantha42)

Ram Kumar

Their dedication, expertise, and valuable contributions have greatly contributed to the success of this project. We appreciate their hard work and commitment to excellence.
# License
This project is licensed under the MIT License.

