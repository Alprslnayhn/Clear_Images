# clear-images

This project, I updated some of the libraries used in the code to their latest versions as they were outdated. Additionally, I replaced unsupported libraries with more up-to-date ones. These changes have enhanced the overall performance and reliability of the code.

You can access the original version of the code from here.
https://github.com/aditya9211/Blur-and-Clear-Classification


# Python3+ user install Tkinter package (Python 3.5.xx)
# Currently code is supported for Python 3.5.xx version only
sudo apt-get install python3-tk
# Clone the repo
git clone https://github.com/aditya9211/Blur-and-Clear-Classification.git
# Change the working Directory
cd Blur-and-Clear-Classification/
# Install the requirements
pip install -r requirements.txt
# Train the Network
python3 train.py  --good_path  '/home/......good/'  --bad_path  '/home/......./bad/'
# Test the Network 
python3 test.py
# Predict output 
python3 predict.py --img '/home/....../laptop.png'
