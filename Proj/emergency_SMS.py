import pywhatkit as kit
import sys
import requests
import time
import os

def get_user_location():
    """Fetch user's location based on IP."""
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        if data["status"] == "success":
            location = f"Location: {data['city']}, {data['regionName']}, {data['country']} (Lat: {data['lat']}, Lon: {data['lon']})"
            return location
        else:
            return "Location not available"
    except Exception as e:
        return f"Error fetching location: {e}"

def close_whatsapp_web():
    """Automatically closes WhatsApp Web by terminating the browser process."""
    time.sleep(10)  # Wait for the message to be sent
    
    # Check for browsers commonly used to open WhatsApp Web
    browsers = [ "msedge.exe", "firefox.exe", "opera.exe"]

    for browser in browsers:
        try:
            os.system(f"taskkill /IM {browser} /F")
            print(f"{browser} closed successfully!")
        except Exception as e:
            print(f"Error closing {browser}: {e}")

def send_sos_whatsapp():
    """Sends emergency message via WhatsApp and closes WhatsApp Web."""
    phone_number = "+918360912024"  # Replace with actual emergency contact number
    location = get_user_location()
    message = f"🚨 EMERGENCY ALERT! 🚨\nPlease check immediately.\n\n{location}"
    
    try:
        kit.sendwhatmsg_instantly(phone_number, message)
        print("Emergency WhatsApp message sent successfully!")

        # Close WhatsApp Web after sending message
        close_whatsapp_web()

    except Exception as e:
        print(f"Failed to send message: {e}")


