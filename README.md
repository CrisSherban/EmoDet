# EmoDet
Emotion Detection with TensorFlow Light for RaspberryPi 4 using Python and Django 

<img src= "/DemoPics/rasp1.jpg">

## Usage
*   Depending on the available mobile device, in order to configure it 
    as a Django Development Server that runs this software you will have
    to install the following packages and libraries:
    * django
    * numpy
    * opencv
    * tensorflow lite
    * pathlib
    * mss
    * pyautogui
    
    Most of them can be installed through PIP if you are using a RaspberryPI
    with ArchLinux or ManjaroLinux (aarch64 version).
    The Tensorflow Light wheel has to be built (to this date) if you want to 
    use python3.8.
    
#### How to star a server with Django:
*   cd into EmoDet folder where manage.py is located
*   Disable DHCP on your connection with the router and set a 
    static IP, for example: `192.168.1.170`
*   Simply run `python manage.py runserver 192.168.1.170:8000`
*   If you want to use a different port just remember to add it to 
    `EmoDet/settings.py`.
  
#### How to use the server:
*   Open a browser and type `192.168.1.170:8000`
*   Choose the desired acquisition type
*   Enjoy

<img src= "/DemoPics/demo.png">
