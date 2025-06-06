import geocoder
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

class GPSService:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="voice_assistant")
        self.last_location = None

    def get_current_gps_location(self):
        """Get precise GPS coordinates using device GPS"""
        try:
            # Get raw GPS coordinates
            g = geocoder.ip('me')
            if not g.latlng:
                g = geocoder.gps()
            
            if g.latlng:
                lat, lon = g.latlng
                location = self.geolocator.reverse(f"{lat}, {lon}")
                
                self.last_location = {
                    'coordinates': (lat, lon),
                    'address': location.address,
                    'city': location.raw.get('address', {}).get('city', 'Unknown'),
                    'region': location.raw.get('address', {}).get('state', 'Unknown'),
                    'country': location.raw.get('address', {}).get('country', 'Unknown'),
                    'postal': location.raw.get('address', {}).get('postcode', 'Unknown')
                }
                return self.last_location
                
        except Exception as e:
            print(f"GPS Error: {e}")
        return None

    def get_distance_from(self, lat, lon):
        """Calculate distance from given coordinates to current location"""
        if not self.last_location:
            self.get_current_gps_location()
            
        if not self.last_location:
            return None
            
        try:
            current_coords = self.last_location['coordinates']
            return geodesic(current_coords, (lat, lon)).km
        except Exception as e:
            print(f"Distance calculation error: {e}")
        return None

    def get_address_from_coords(self, lat, lon):
        """Get human-readable address from GPS coordinates"""
        try:
            location = self.geolocator.reverse(f"{lat}, {lon}")
            return location.address
        except Exception as e:
            print(f"Reverse geocoding error: {e}")
        return None

if __name__ == "__main__":
    gps = GPSService()
    location = gps.get_current_gps_location()
    if location:
        print("GPS Coordinates:", location['coordinates'])
        print("Full Address:", location['address'])
        print("City:", location['city'])
        print("Region:", location['region'])
        print("Country:", location['country'])
    else:
        print("Could not get GPS location")




