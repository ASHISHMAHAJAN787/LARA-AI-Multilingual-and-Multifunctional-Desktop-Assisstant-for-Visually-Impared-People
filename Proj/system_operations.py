import os
import webbrowser
import speedtest
import subprocess

class SystemTasks:
    def open_website(self, query):
        """
        Open a website based on the given query.
        Dynamically constructs the URL if not explicitly provided.
        """
        try:
            # Check if the query already contains a domain (e.g., google.com)
            if "." in query:
                url = query if query.startswith("http") else f"https://{query}"
            else:
                # Assume it's a common name and construct the URL dynamically
                url = f"https://{query}.com"
            
            # Open the website in the default browser
            os.system(f"start {url}")
            return f"Opening {query}"
        except Exception as e:
            return f"Sorry, I couldn't open the website. Error: {e}"

    def open_app(self, app_name):
        """
        Open a desktop application based on the given name.
        """
        app_paths = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "command prompt": "cmd.exe",
            "paint": "mspaint.exe",
            "MS word":"winword.exe",
            "ms word":"winword.exe",
            "word": "winword.exe",
            "excel": "excel.exe",
            "powerpoint": "powerpnt.exe",
            "visual studio code": "code",
            "chrome": "chrome.exe",
            "brave": "brave.exe",
            # Add paths to other applications as needed
        }

        # Find the app path and open it
        app_path = app_paths.get(app_name.lower())
        if app_path:
            os.system(f"start {app_path}")
            return f"Opening {app_name}"
        else:
            return f"Sorry, I don't know how to open {app_name}."



    def take_screenshot(self):
        os.system("snippingtool")

def close_window(app_name=None):
    """
    Close an application by its process name.
    If no application name is provided, closes a default application.
    """
    try:
        if app_name:
            os.system(f"taskkill /f /im {app_name}.exe")
            return f"Closing {app_name}"
        else:
            return "Please specify the application to close."
    except Exception as e:
        return f"Error closing the application: {e}"


    def check_internet_speed(self):
        """
        Check the current internet speed (download and upload).
        """
        try:
            st = speedtest.Speedtest()
            st.get_best_server()
            download_speed = st.download() / 1_000_000  # Convert to Mbps
            upload_speed = st.upload() / 1_000_000  # Convert to Mbps
            return f"Download speed: {download_speed:.2f} Mbps, Upload speed: {upload_speed:.2f} Mbps"
        except Exception as e:
            return f"Failed to check internet speed. Error: {e}"