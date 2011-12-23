rm src/*.c *.so -rf build
python setup.py build
sudo python setupegg.py develop
