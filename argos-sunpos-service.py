#!/usr/bin/env python3
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pvlib import solarposition, irradiance, atmosphere, clearsky
from pvlib.location import Location
import json
import socket
import os
import struct
import threading
from pathlib import Path
import requests
import math

# CRITICAL: Check if sun actually rises above horizon
# -0.833° is the internationally accepted standard geometric altitude for civil sunrise/sunset,
# accounting for both atmospheric refraction and the sun's physical size.
APPARENT_SUNRISE_ALTITUDE = -0.833  # degrees

# Compare to https://climate-adapt.eea.europa.eu/en/observatory/evidence/maps-and-charts/cams-uv-index-forecast

def estimate_ozone_column(latitude, day_of_year):
    """
    Estimate total ozone column in Dobson Units (DU) based on latitude and day of year.
    Based on Van Heuklon (1979) simplified approximation with corrections.
    """
    lat_rad = math.radians(abs(latitude))

    # Annual mean ozone by latitude
    if abs(latitude) < 20:
        base_ozone = 270  # Tropical (increased from 260)
    elif abs(latitude) < 40:
        base_ozone = 295  # Subtropical (increased from 290)
    elif abs(latitude) < 60:
        base_ozone = 330  # Mid-latitude
    else:
        base_ozone = 370  # Polar

    # Seasonal variation (max in spring, min in fall for each hemisphere)
    # Day 80 = ~March 21, Day 264 = ~September 21
    if latitude >= 0:  # Northern hemisphere
        phase = day_of_year - 80  # Peak in NH spring
    else:  # Southern hemisphere
        phase = day_of_year - 264  # Peak in SH spring

    # CORRECTED: Seasonal amplitude - much smaller near equator
    # Use latitude^1.0 instead of latitude^0.5 to reduce equatorial variation
    latitude_factor = (abs(latitude) / 90.0) ** 1.0  # Changed from 0.5
    amplitude = 50 * latitude_factor

    # Additional damping for very low latitudes (tropics are VERY stable)
    if abs(latitude) < 10:
        amplitude *= 0.3  # Reduce to 30% for near-equator

    seasonal = amplitude * math.cos(2 * math.pi * phase / 365.25)

    ozone_du = base_ozone + seasonal

    return max(200, min(500, ozone_du))  # Clamp to reasonable range


def calculate_uv_transmission(solar_zenith_angle, ozone_du, airmass):
    """
    Calculate UV transmission through ozone layer.
    Uses empirically validated absorption for erythemal UV.
    """
    if solar_zenith_angle >= 90:
        return 0.0

    # Effective ozone path (slant path correction)
    ozone_path = ozone_du * airmass

    # CORRECTED absorption coefficients based on empirical UV measurements
    # These are tuned to match ground-based UV monitoring networks
    #
    # UV-B (280-315nm) - primary component of erythemal UV
    # Reduced from 0.15 to match empirical observations
    k_uvb = 0.08  # Per 100 DU (empirically validated)
    transmission_uvb = math.exp(-k_uvb * ozone_path / 100.0)

    # UV-A (315-400nm) - minor erythemal contribution
    k_uva = 0.01  # Per 100 DU
    transmission_uva = math.exp(-k_uva * ozone_path / 100.0)

    # Erythemal action spectrum weighting
    # UV-B contributes ~75% to erythemal dose at surface
    erythemal_transmission = 0.75 * transmission_uvb + 0.25 * transmission_uva

    return erythemal_transmission


def calculate_current_uv_index(solar_elevation, clearsky_ghi, clearsky_dni,
                               airmass, latitude, day_of_year, aod_550nm=0.1):
    """
    Calculate current UV Index with empirically tuned parameters.
    """
    # If sun is below horizon, UV Index = 0
    if solar_elevation <= 0:
        return 0.0

    solar_zenith_angle = 90.0 - solar_elevation

    # 1. Estimate climatological ozone
    ozone_du = estimate_ozone_column(latitude, day_of_year)

    # 2. Calculate UV transmission through ozone
    uv_transmission = calculate_uv_transmission(solar_zenith_angle, ozone_du, airmass)

    # 3. TUNED: Base erythemal fraction
    # From empirical UV monitoring data (WMO/WHO networks):
    # At solar noon (elevation ~60-70°), clear sky: UV/GHI ≈ 0.04-0.05%
    # This is for overhead sun with minimal atmospheric path
    base_erythemal_fraction = 0.00038  # 0.038% at reference conditions

    # 4. Solar elevation adjustment
    # UV increases non-linearly with sun angle
    # But less aggressive than before to avoid over-attenuation at low angles
    sin_elevation = math.sin(math.radians(solar_elevation))

    # Use moderate power law - empirically matched to UV monitoring data
    elevation_factor = sin_elevation ** 0.8  # Less aggressive than 1.2
    # Minimum factor to prevent over-attenuation at low sun
    elevation_factor = max(0.15, elevation_factor)

    # Effective fraction adjusted for sun angle
    effective_fraction = base_erythemal_fraction * elevation_factor

    # 5. Calculate base erythemal irradiance from GHI
    uv_irradiance_base = clearsky_ghi * effective_fraction

    # 6. Apply ozone transmission
    uv_irradiance_ozone = uv_irradiance_base * uv_transmission

    # 7. Aerosol scattering
    # UV aerosol optical depth (Ångström relationship)
    aod_uv = atmosphere.angstrom_aod_at_lambda(
        aod0=aod_550nm,
        lambda0=550,
        lambda1=310,  # UV wavelength ~310nm
        alpha=1.3  # Typical Ångström exponent for aerosols
    )

    # Aerosol transmission (simpler formula for clear sky)
    # Reduced attenuation coefficient from 0.8 to 0.5 (more typical for UV scattering)
    aerosol_transmission = math.exp(-0.5 * aod_uv * airmass)

    # 8. Apply aerosol correction
    uv_irradiance_corrected = uv_irradiance_ozone * aerosol_transmission

    # 9. Surface albedo (minimal for typical ground)
    albedo_boost = 1.02  # Reduced from 1.03 (grass/soil UV albedo ~2-3%)
    uv_irradiance_final = uv_irradiance_corrected * albedo_boost

    # 10. Convert to UV Index
    uv_index = uv_irradiance_final / 0.025

    #print(f"DEBUG: Elevation={solar_elevation:.1f}°, Airmass={airmass:.2f}, Ozone={ozone_du:.0f} DU")
    #print(f"DEBUG: UV transmission={uv_transmission:.3f}, Aerosol transmission={aerosol_transmission:.3f}")
    #print(f"DEBUG: Elevation factor={elevation_factor:.3f}, Effective fraction={effective_fraction:.6f}")
    #print(f"DEBUG: Base UV={uv_irradiance_base:.4f} W/m², Final={uv_irradiance_final:.4f} W/m²")
    #print(f"DEBUG: UV Index = {uv_index:.2f}")

    # Clamp to reasonable range
    return max(0.0, min(uv_index, 16.0))


def get_airmass_absolute(location, time):
    """
    Calculate absolute airmass (pressure-corrected) for a given time.

    Args:
        location: pvlib Location object
        time: pandas Timestamp or DatetimeIndex

    Returns:
        float: Absolute airmass value, or 1.0 if invalid/unavailable
    """
    # Ensure time is a DatetimeIndex
    if not isinstance(time, pd.DatetimeIndex):
        time = pd.DatetimeIndex([time])

    airmass = location.get_airmass(time, model='kastenyoung1989')

    # Use airmass_absolute (accounts for pressure/altitude)
    if 'airmass_absolute' in airmass:
        airmass_value = float(airmass['airmass_absolute'].iloc[0])

        # Validate: airmass_absolute can be < 1.0 at high altitude
        # Valid range: 0.01 (extreme altitude) to 40 (extreme zenith angle)
        if math.isfinite(airmass_value) and 0.01 <= airmass_value <= 40:
            return airmass_value
        else:
            # Invalid value (NaN, inf, or out of physical range)
            #print(f"WARNING: Invalid airmass_absolute={airmass_value}, using fallback 1.0")
            return 1.0
    else:
        # No airmass_absolute column, try airmass_relative as fallback
        if 'airmass_relative' in airmass:
            #print(f"WARNING: airmass_absolute not available, using airmass_relative")
            return float(airmass['airmass_relative'].iloc[0])
        else:
            return 1.0


class SolarDataCache:
    def __init__(self):
        self.config_file = Path.home() / '.config' / 'argos-sunpos.json'
        self.atm_cache_file = Path.home() / '.cache' / 'argos-sunpos-atmosphere.json'
        self.config = None
        self.config_mtime = None
        self.atm_cache = None
        self.lock = threading.Lock()
        self.last_printed_date = None

        # Ensure cache directory exists
        self.atm_cache_file.parent.mkdir(parents=True, exist_ok=True)

        self.load_config()
        self.load_atmosphere_cache()

    def find_polar_end_date(self, start_date, is_polar_day):
        """Find when polar day or polar night ends (up to 180 days ahead)"""
        # Ensure start_date is a timezone-aware pandas Timestamp
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.Timestamp(start_date)

        for days_ahead in range(1, 181):
            future_date = start_date + pd.Timedelta(days=days_ahead)
            future_time_series = pd.DatetimeIndex([future_date])

            future_transit = solarposition.sun_rise_set_transit_spa(
                times=future_time_series,
                latitude=self.config['latitude'],
                longitude=self.config['longitude'],
                delta_t=67.0
            )

            future_sunrise_utc = future_transit['sunrise'].iloc[0]
            future_sunset_utc = future_transit['sunset'].iloc[0]

            # Get max altitude for verification
            future_transit_utc = future_transit['transit'].iloc[0]
            if pd.isna(future_transit_utc):
                continue

            future_noon_pos = solarposition.spa_python(
                time=pd.DatetimeIndex([future_transit_utc]),
                latitude=self.config['latitude'],
                longitude=self.config['longitude'],
                altitude=self.config['altitude']
            )
            future_max_altitude = float(future_noon_pos['apparent_elevation'].iloc[0])

            # Check if condition has changed
            if is_polar_day:
                # Polar day ends when BOTH sunrise and sunset occur (normal day/night cycle returns)
                if (not pd.isna(future_sunset_utc) and
                    not pd.isna(future_sunrise_utc) and
                    future_max_altitude > APPARENT_SUNRISE_ALTITUDE):
                    return days_ahead, pd.Timestamp(future_date)
            else:
                # Polar night ends when sun can rise (max_altitude rises enough)
                if future_max_altitude > APPARENT_SUNRISE_ALTITUDE:
                    return days_ahead, pd.Timestamp(future_date)

        return None, None

    def load_config(self):
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    config = json.load(f)

                # Validate altitude (reasonable range: Dead Sea to high aviation)
                altitude = config.get('altitude', 0)
                MAX_ALTITUDE = 15000  # Conservative limit (formula breaks at 44,331.514m)
                if altitude < -500 or altitude > MAX_ALTITUDE:
                    print(f"WARNING: Altitude {altitude}m is outside reasonable range (-500 to {MAX_ALTITUDE}m)")
                    print(f"         Clamping to valid range. Clearsky models are designed for troposphere.")
                    config['altitude'] = max(-500, min(MAX_ALTITUDE, altitude))

                # Ensure 'offline' key exists (default to True if missing)
                if 'offline' not in config:
                    config['offline'] = True

                self.config = config
                self.config_mtime = self.config_file.stat().st_mtime
                offline_status = "OFFLINE" if self.config['offline'] else "ONLINE"
                print(f"Loaded config: lat={self.config['latitude']}, lon={self.config['longitude']}, tz={self.config['timezone']}, mode={offline_status}")
            except Exception as e:
                print(f"Error loading config: {e}")
                self._set_default_config()
        else:
            self._set_default_config()

    def _set_default_config(self):
        """Set default configuration"""
        self.config = {
            'latitude': 37.23,
            'longitude': -115.81,
            'altitude': 2000,
            'timezone': 'America/Los_Angeles',
            'offline': True  # Default to offline mode (privacy + no internet needed)
        }
        self.config_mtime = None
        print("Using default configuration")

    def check_config_changed(self):
        """Check if config file has been modified and reload if necessary"""
        if not self.config_file.exists():
            if self.config_mtime is not None:
                print("Config file deleted, reverting to defaults")
                self._set_default_config()
                return True
            return False

        try:
            current_mtime = self.config_file.stat().st_mtime
            if self.config_mtime is None or current_mtime > self.config_mtime:
                print(f"Config file changed (mtime: {current_mtime}), reloading...")
                self.load_config()
                return True
        except Exception as e:
            print(f"Error checking config file: {e}")

        return False

    def truncate_coordinates(self, lat, lon, decimals=1):
        """Truncate coordinates for privacy (1 decimal ≈ 11km grid)"""
        return round(lat, decimals), round(lon, decimals)

    def load_atmosphere_cache(self):
        """Load cached atmospheric data"""
        if self.atm_cache_file.exists():
            try:
                with open(self.atm_cache_file) as f:
                    self.atm_cache = json.load(f)
                print(f"Loaded atmosphere cache from {self.atm_cache_file}")
            except Exception as e:
                print(f"Error loading atmosphere cache: {e}")
                self.atm_cache = None
        else:
            self.atm_cache = None

    def is_atmosphere_cache_valid(self, truncated_lat, truncated_lon, query_date):
        """Check if cached atmospheric data is still valid (< 12 hours old, same location, same date)"""
        if not self.atm_cache:
            return False

        try:
            cache_time = datetime.fromisoformat(self.atm_cache['timestamp'])
            age_hours = (datetime.now(timezone.utc) - cache_time).total_seconds() / 3600

            # Check if cache is less than 12 hours old
            if age_hours > 12:
                return False

            # Check if location matches (truncated)
            if (self.atm_cache.get('latitude') != truncated_lat or
                self.atm_cache.get('longitude') != truncated_lon):
                return False

            # Check if the query date matches (important for daily data)
            if self.atm_cache.get('query_date') != query_date:
                return False

            return True
        except Exception as e:
            print(f"Error validating cache: {e}")
            return False

    def fetch_nasa_power_data(self, lat, lon):
        """Fetch atmospheric data from NASA POWER API"""
        # Truncate coordinates for privacy (1 decimal = ~11km grid)
        truncated_lat, truncated_lon = self.truncate_coordinates(lat, lon, decimals=1)

        # NASA POWER has 1-2 day lag
        daybefore = datetime.now(timezone.utc) - pd.Timedelta(days=3)
        query_date = daybefore.strftime('%Y%m%d')

        # Check if we have valid cached data for this location and date
        if self.is_atmosphere_cache_valid(truncated_lat, truncated_lon, query_date):
            if query_date != self.last_printed_date:
                print(f"Using cached atmospheric data (date: {query_date})")
                self.last_printed_date = query_date
            return self.atm_cache['data']

        print(f"Fetching fresh atmospheric data for lat={truncated_lat}, lon={truncated_lon}, date={query_date}")

        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            'parameters': 'T2M,RH2M,PS,AOD_55,ALLSKY_SFC_UV_INDEX',  # Temperature, Humidity, Pressure, Aerosol Optical Depth
            'community': 'RE',
            'longitude': truncated_lon,
            'latitude': truncated_lat,
            'start': query_date,
            'end': query_date,
            'format': 'JSON'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract values for the queried date
            properties = data['properties']['parameter']

            # Get values, use defaults if missing (-999 indicates no data)
            temp = properties.get('T2M', {}).get(query_date, -999)
            humidity = properties.get('RH2M', {}).get(query_date, -999)
            pressure = properties.get('PS', {}).get(query_date, -999)
            aod_550 = properties.get('AOD_55', {}).get(query_date, -999)
            uv_index = properties.get('ALLSKY_SFC_UV_INDEX', {}).get(query_date, -999)

            print(f"DEBUG: Raw NASA POWER values for {query_date}: T={temp}, RH={humidity}, P={pressure}, AOD={aod_550}, UV={uv_index}")

            # Check if we got valid data
            if temp == -999 and humidity == -999 and pressure == -999 and aod_550 == -999:
                print("WARNING: All NASA POWER values are -999 (no data available), using defaults")

            atm_data = {
                'temperature': temp if temp != -999 else 12,  # Default 12°C
                'humidity': humidity if humidity != -999 else 50,  # Default 50%
                'pressure': pressure * 1000 if pressure != -999 else 101325,  # Convert kPa to Pa
                'aod_550nm': aod_550 if aod_550 != -999 else 0.1,  # Aerosol Optical Depth (clean atmosphere default)
                'uv_index_max': uv_index if uv_index != -999 else None
            }

            # Cache the results with query date
            self.atm_cache = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'query_date': query_date,
                'latitude': truncated_lat,
                'longitude': truncated_lon,
                'data': atm_data
            }

            # Save cache to file
            try:
                with open(self.atm_cache_file, 'w') as f:
                    json.dump(self.atm_cache, f, indent=2)
                uv_str = f"{atm_data['uv_index_max']:.1f}" if atm_data.get('uv_index_max') is not None else "N/A"
                print(f"Cached atmospheric data (date={query_date}): T={atm_data['temperature']:.1f}°C, RH={atm_data['humidity']:.0f}%, AOD={atm_data['aod_550nm']:.3f}, UV={uv_str}")
            except Exception as e:
                print(f"Error saving atmosphere cache: {e}")

            return atm_data

        except requests.exceptions.Timeout:
            print("NASA POWER API timeout - using defaults")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching NASA POWER data: {e}")
        except Exception as e:
            print(f"Unexpected error with NASA POWER: {e}")

        # Return defaults if fetch failed
        return {
            'temperature': 12,
            'humidity': 50,
            'pressure': 101325,
            'aod_550nm': 0.1,
            'uv_index_max': None
        }

    def calculate_solar_data(self):
        """Calculate solar position - called on every query for real-time accuracy"""
        try:
            now_utc = datetime.now(timezone.utc)
            local_tz = ZoneInfo(self.config['timezone'])
            current_time_series = pd.DatetimeIndex([now_utc])

            uv_index_zenith = None

            solar_pos = solarposition.spa_python(
                time=current_time_series,
                latitude=self.config['latitude'],
                longitude=self.config['longitude'],
                altitude=self.config['altitude'],
                pressure=101325,
                temperature=12
            )

            transit_times = solarposition.sun_rise_set_transit_spa(
                times=current_time_series,
                latitude=self.config['latitude'],
                longitude=self.config['longitude'],
                delta_t=67.0
            )

            result = {
                'current_altitude': float(solar_pos['apparent_elevation'].iloc[0]),
                'current_azimuth': float(solar_pos['azimuth'].iloc[0]),
                'timestamp': now_utc.isoformat(),
                'config': {
                    'latitude': self.config['latitude'],
                    'longitude': self.config['longitude'],
                    'altitude': self.config['altitude'],
                    'timezone': self.config['timezone']
                }
            }

            # Calculate solar irradiance using clear-sky model
            try:
                location = Location(
                    latitude=self.config['latitude'],
                    longitude=self.config['longitude'],
                    altitude=self.config['altitude'],
                    tz=self.config['timezone']
                )

                # Check offline mode
                offline_mode = self.config.get('offline', True)

                if offline_mode:
                    # OFFLINE MODE: Use Ineichen with lookup Linke turbidity
                    # No internet required, good accuracy (±10-15%)
                    linke_turbidity = clearsky.lookup_linke_turbidity(
                        time=current_time_series,
                        latitude=self.config['latitude'],
                        longitude=self.config['longitude']
                    )
                    clearsky_data = location.get_clearsky(
                        current_time_series,
                        model='ineichen',
                        linke_turbidity=linke_turbidity
                    )
                    model_used = 'ineichen_offline_with_linke_lookup'
                    #print(f"DEBUG: Using Linke turbidity = {linke_turbidity.iloc[0]:.2f}")
                    atm_data = None
                else:
                    # ONLINE MODE: Fetch NASA POWER data and use Simplified Solis
                    # Better accuracy (±5-10%) using real atmospheric data
                    atm_data = self.fetch_nasa_power_data(
                        self.config['latitude'],
                        self.config['longitude']
                    )

                    uv_str = f"{atm_data['uv_index_max']:.1f}" if atm_data.get('uv_index_max') is not None else "N/A"
                    #print(f"DEBUG: Atmospheric data received: T={atm_data['temperature']:.1f}°C, "
                    #      f"RH={atm_data['humidity']:.1f}%, AOD={atm_data['aod_550nm']:.4f}, UV={uv_str}")

                    # Calculate precipitable water from temperature and humidity
                    precipitable_water = atmosphere.gueymard94_pw(
                        temp_air=atm_data['temperature'],
                        relative_humidity=atm_data['humidity']
                    )

                    try:
                        # Use Simplified Solis model with atmospheric data
                        # Convert AOD from 550nm to 700nm using pvlib
                        aod700 = atmosphere.angstrom_aod_at_lambda(
                            aod0=atm_data['aod_550nm'],
                            lambda0=550,
                            lambda1=700,
                            alpha=1.3
                        )

                        #print(f"DEBUG: Attempting Simplified Solis with aod700={aod700:.4f}, pw={precipitable_water:.2f}")

                        clearsky_data = location.get_clearsky(
                            current_time_series,
                            model='simplified_solis',
                            aod700=aod700,
                            precipitable_water=precipitable_water
                        )
                        model_used = 'simplified_solis_online'
                    except Exception as e:
                        print(f"DEBUG: Simplified Solis failed with error: {type(e).__name__}: {e}")
                        import traceback
                        print("DEBUG: Full traceback:")
                        traceback.print_exc()
                        print("DEBUG: Falling back to Ineichen model with Linke turbidity lookup")

                        # Fallback with Linke turbidity lookup
                        linke_turbidity = clearsky.lookup_linke_turbidity(
                            time=current_time_series,
                            latitude=self.config['latitude'],
                            longitude=self.config['longitude']
                        )
                        clearsky_data = location.get_clearsky(
                            current_time_series,
                            model='ineichen',
                            linke_turbidity=linke_turbidity
                        )
                        model_used = 'ineichen_fallback_with_linke_lookup'

                # Calculate extraterrestrial irradiance for reference
                dni_extra = irradiance.get_extra_radiation(current_time_series)
                # Get airmass for additional context
                airmass_value = get_airmass_absolute(location, current_time_series)

                # Calculate current UV Index
                current_elevation = float(solar_pos['apparent_elevation'].iloc[0])
                current_ghi = float(clearsky_data['ghi'].iloc[0])
                current_dni = float(clearsky_data['dni'].iloc[0])

                # Get AOD if in online mode
                current_aod = atm_data['aod_550nm'] if atm_data else 0.1

                # Get day of year
                now_utc = datetime.now(timezone.utc)
                day_of_year = now_utc.timetuple().tm_yday

                # Calculate UV Index with new function
                uv_index_current = calculate_current_uv_index(
                    solar_elevation=current_elevation,
                    clearsky_ghi=current_ghi,
                    clearsky_dni=current_dni,
                    airmass=airmass_value,
                    latitude=self.config['latitude'],
                    day_of_year=day_of_year,
                    aod_550nm=current_aod
                )

                result['irradiance'] = {
                    'ghi': float(clearsky_data['ghi'].iloc[0]),  # Global Horizontal Irradiance (W/m²)
                    'dni': float(clearsky_data['dni'].iloc[0]),  # Direct Normal Irradiance (W/m²)
                    'dhi': float(clearsky_data['dhi'].iloc[0]),  # Diffuse Horizontal Irradiance (W/m²)
                    'dni_extra': float(dni_extra.iloc[0]),  # Extraterrestrial DNI (W/m²)
                    'airmass': airmass_value,
                    'uv_index_current': round(uv_index_current, 1),  # Add current UV Index
                    'uv_index_zenith': round(uv_index_zenith, 1) if uv_index_zenith is not None else None,  # Max UV Index (at solar noon)
                    'model': model_used,
                    'offline_mode': offline_mode
                }

                # Add Linke turbidity to result if in offline mode
                if offline_mode:
                    result['irradiance']['linke_turbidity'] = float(linke_turbidity.iloc[0])

                # Add atmospheric data if in online mode
                if atm_data:
                    result['irradiance']['atmospheric'] = {
                        'temperature': atm_data['temperature'],
                        'humidity': atm_data['humidity'],
                        'pressure': atm_data['pressure'],
                        'aod_550nm': atm_data['aod_550nm'],
                        'uv_index_max': atm_data['uv_index_max']
                    }

                # Add note if sun is below horizon
                if solar_pos['apparent_elevation'].iloc[0] < 0:
                    result['irradiance']['note'] = 'Sun below horizon (nighttime)'

            except Exception as e:
                print(f"Error calculating irradiance: {e}")
                result['irradiance'] = {
                    'error': str(e),
                    'ghi': 0.0,
                    'dni': 0.0,
                    'dhi': 0.0
                }

            # Calculate solar noon and max altitude FIRST
            transit_utc = transit_times['transit'].iloc[0]
            max_altitude = None
            uv_index_zenith = None

            if not pd.isna(transit_utc):
                transit_local = transit_utc.tz_convert(local_tz)
                noon_pos = solarposition.spa_python(
                    time=pd.DatetimeIndex([transit_utc]),
                    latitude=self.config['latitude'],
                    longitude=self.config['longitude'],
                    altitude=self.config['altitude']
                )
                max_altitude = float(noon_pos['apparent_elevation'].iloc[0])

                # Calculate UV index at zenith (solar noon)
                if max_altitude > 0:  # Only if sun is above horizon
                    try:
                        # Get clear-sky irradiance at solar noon
                        if offline_mode:
                            linke_turbidity_noon = clearsky.lookup_linke_turbidity(
                                time=pd.DatetimeIndex([transit_utc]),
                                latitude=self.config['latitude'],
                                longitude=self.config['longitude']
                            )
                            clearsky_noon = location.get_clearsky(
                                pd.DatetimeIndex([transit_utc]),
                                model='ineichen',
                                linke_turbidity=linke_turbidity_noon
                            )
                        else:
                            # Use same atmospheric data as current calculation
                            if 'clearsky_data' in locals() and model_used == 'simplified_solis_online':
                                # Use Simplified Solis for noon as well
                                clearsky_noon = location.get_clearsky(
                                    pd.DatetimeIndex([transit_utc]),
                                    model='simplified_solis',
                                    aod700=aod700,
                                    precipitable_water=precipitable_water
                                )
                            else:
                                # Fallback to Ineichen
                                linke_turbidity_noon = clearsky.lookup_linke_turbidity(
                                    time=pd.DatetimeIndex([transit_utc]),
                                    latitude=self.config['latitude'],
                                    longitude=self.config['longitude']
                                )
                                clearsky_noon = location.get_clearsky(
                                    pd.DatetimeIndex([transit_utc]),
                                    model='ineichen',
                                    linke_turbidity=linke_turbidity_noon
                                )

                        airmass_noon_value = get_airmass_absolute(location, transit_utc)

                        # Get AOD (use same as current)
                        aod_noon = atm_data['aod_550nm'] if atm_data else 0.1

                        # Calculate UV Index at zenith
                        uv_index_zenith = calculate_current_uv_index(
                            solar_elevation=max_altitude,
                            clearsky_ghi=float(clearsky_noon['ghi'].iloc[0]),
                            clearsky_dni=float(clearsky_noon['dni'].iloc[0]),
                            airmass=airmass_noon_value,
                            latitude=self.config['latitude'],
                            day_of_year=day_of_year,
                            aod_550nm=aod_noon
                        )

                        # Update the irradiance dictionary with the calculated zenith UV
                        if 'irradiance' in result and 'error' not in result['irradiance']:
                            result['irradiance']['uv_index_zenith'] = round(uv_index_zenith, 1)

                    except Exception as e:
                        print(f"Error calculating UV index at zenith: {e}")
                        uv_index_zenith = None

                result['solar_noon'] = {
                    'time': transit_local.strftime('%H:%M:%S'),
                    'max_altitude': max_altitude,
                }
            else:
                result['solar_noon'] = None

            has_actual_sunrise = max_altitude is not None and max_altitude > APPARENT_SUNRISE_ALTITUDE

            # Sunrise
            sunrise_utc = transit_times['sunrise'].iloc[0]
            sunrise_local = None

            if pd.isna(sunrise_utc) or not has_actual_sunrise:
                result['sunrise'] = None
                if max_altitude is not None and max_altitude <= APPARENT_SUNRISE_ALTITUDE:
                    result['sunrise_note'] = f"No sunrise (max alt: {max_altitude:.2f}°)"
            else:
                sunrise_local = sunrise_utc.tz_convert(local_tz)
                sunrise_pos = solarposition.spa_python(
                    time=pd.DatetimeIndex([sunrise_utc]),
                    latitude=self.config['latitude'],
                    longitude=self.config['longitude'],
                    altitude=self.config['altitude']
                )
                result['sunrise'] = {
                    'time': sunrise_local.strftime('%H:%M:%S'),
                    'azimuth': float(sunrise_pos['azimuth'].iloc[0])
                }

            # Sunset
            sunset_utc = transit_times['sunset'].iloc[0]
            sunset_local = None

            if pd.isna(sunset_utc) or not has_actual_sunrise:
                result['sunset'] = None
                if max_altitude is not None and max_altitude <= APPARENT_SUNRISE_ALTITUDE:
                    result['sunset_note'] = f"No sunset (max alt: {max_altitude:.2f}°)"
            else:
                sunset_local = sunset_utc.tz_convert(local_tz)
                sunset_pos = solarposition.spa_python(
                    time=pd.DatetimeIndex([sunset_utc]),
                    latitude=self.config['latitude'],
                    longitude=self.config['longitude'],
                    altitude=self.config['altitude']
                )
                result['sunset'] = {
                    'time': sunset_local.strftime('%H:%M:%S'),
                    'azimuth': float(sunset_pos['azimuth'].iloc[0])
                }

            # Add day type classification
            is_polar_day = (pd.isna(sunrise_utc) or pd.isna(sunset_utc)) and has_actual_sunrise

            if is_polar_day:
                result['day_type'] = 'polar_day'
            elif max_altitude is None:
                result['day_type'] = 'unknown'
            elif max_altitude > 6.0:
                result['day_type'] = 'normal'
            elif max_altitude > APPARENT_SUNRISE_ALTITUDE:
                result['day_type'] = 'short_day'
            elif max_altitude > -6.0:
                result['day_type'] = 'polar_night_civil_twilight'
            elif max_altitude > -12.0:
                result['day_type'] = 'polar_night_nautical_twilight'
            elif max_altitude > -18.0:
                result['day_type'] = 'polar_night_astronomical_twilight'
            else:
                result['day_type'] = 'polar_night_total'

            # Handle polar day (24 hours of daylight)
            if result['day_type'] == 'polar_day':
                result['daylight_duration'] = {
                    'seconds': 86400,
                    'formatted': '24h 0m 0s'
                }
                days, end_date = self.find_polar_end_date(now_utc, is_polar_day=True)
                if days:
                    end_date_local = end_date.tz_convert(local_tz)
                    result['polar_ends'] = {
                        'days': days,
                        'date': end_date_local.strftime('%Y-%m-%d'),
                        'condition': 'polar_day',
                        'formatted': f"Polar day ends in {days} day{'s' if days != 1 else ''}"
                    }
                else:
                    result['polar_ends'] = {'note': 'Polar day continues beyond 6 months'}

            # Handle polar night variants
            elif result['day_type'].startswith('polar_night'):
                days, end_date = self.find_polar_end_date(now_utc, is_polar_day=False)
                if days:
                    end_date_local = end_date.tz_convert(local_tz)
                    result['polar_ends'] = {
                        'days': days,
                        'date': end_date_local.strftime('%Y-%m-%d'),
                        'condition': 'polar_night',
                        'formatted': f"Polar night ends in {days} day{'s' if days != 1 else ''}"
                    }
                else:
                    result['polar_ends'] = {'note': 'Polar night continues beyond 6 months'}

            # Calculate daylight duration for normal/short days
            elif has_actual_sunrise and sunrise_local and sunset_local:
                # Normal calculation - today's daylight duration
                daylight_duration = (sunset_utc - sunrise_utc).total_seconds()

                # Check if this is a transition day (duration >23 hours or negative)
                if daylight_duration < 0 or daylight_duration > 82800:  # 82800 = 23 hours
                    # This is a transition day - sun set yesterday/rises today, or entering/exiting polar period
                    result['daylight_duration'] = {
                        'note': 'Transition day (entering/exiting polar period)'
                    }
                else:
                    hours = int(daylight_duration // 3600)
                    minutes = int((daylight_duration % 3600) // 60)
                    seconds = int(daylight_duration % 60)

                    result['daylight_duration'] = {
                        'seconds': daylight_duration,
                        'formatted': f"{hours}h {minutes}m {seconds}s"
                    }

                    # Calculate TOMORROW's sunrise/sunset
                    tomorrow_utc = now_utc + pd.Timedelta(days=1)
                    tomorrow_time_series = pd.DatetimeIndex([tomorrow_utc])
                    tomorrow_transit = solarposition.sun_rise_set_transit_spa(
                        times=tomorrow_time_series,
                        latitude=self.config['latitude'],
                        longitude=self.config['longitude'],
                        delta_t=67.0
                    )
                    tomorrow_sunrise_utc = tomorrow_transit['sunrise'].iloc[0]
                    tomorrow_sunset_utc = tomorrow_transit['sunset'].iloc[0]
                    tomorrow_transit_utc = tomorrow_transit['transit'].iloc[0]

                    # Check if tomorrow has sunrise
                    if not pd.isna(tomorrow_sunrise_utc) and not pd.isna(tomorrow_sunset_utc):
                        # Calculate tomorrow's max altitude to verify it's a real sunrise
                        if not pd.isna(tomorrow_transit_utc):
                            tomorrow_noon_pos = solarposition.spa_python(
                                time=pd.DatetimeIndex([tomorrow_transit_utc]),
                                latitude=self.config['latitude'],
                                longitude=self.config['longitude'],
                                altitude=self.config['altitude']
                            )
                            tomorrow_max_altitude = float(tomorrow_noon_pos['apparent_elevation'].iloc[0])

                            if tomorrow_max_altitude > APPARENT_SUNRISE_ALTITUDE:
                                # Tomorrow has actual sunrise
                                tomorrow_daylight = (tomorrow_sunset_utc - tomorrow_sunrise_utc).total_seconds()
                                # Calculate difference
                                difference = tomorrow_daylight - daylight_duration
                                abs_diff = abs(difference)
                                TOLERANCE = 1.0  # seconds
                                diff_minutes = int(abs_diff // 60)
                                diff_seconds = int(abs_diff % 60)

                                if difference > 0:
                                    direction = "longer"
                                elif difference < 0:
                                    direction = "shorter"
                                else:
                                    direction = "same"

                                if abs_diff < TOLERANCE:
                                    # Essentially no change
                                    result['daylight_change'] = {
                                        'seconds': 0.0,
                                        'direction': 'same',
                                        'formatted': 'same length'
                                    }
                                else:
                                    # Significant change
                                    diff_minutes = int(abs_diff // 60)
                                    diff_seconds = int(abs_diff % 60)

                                    if difference > 0:
                                        direction = "longer"
                                    else:
                                        direction = "shorter"

                                    if diff_minutes > 0 and diff_seconds > 0:
                                            formatted = f"{diff_minutes}m {diff_seconds}s {direction}"
                                    elif diff_minutes > 0:
                                        formatted = f"{diff_minutes}m {direction}"
                                    else:
                                        formatted = f"{diff_seconds}s {direction}"

                                    result['daylight_change'] = {
                                        'seconds': difference,
                                        'direction': direction,
                                        'formatted': formatted
                                    }

                            else:
                                # Tomorrow has no sunrise
                                result['daylight_change'] = {
                                    'note': 'No sunrise tomorrow'
                                }
                        else:
                            # Tomorrow has no sunrise/sunset times
                            result['daylight_change'] = {
                                'note': 'No sunrise tomorrow'
                            }
                    else:
                        result['daylight_change'] = {
                            'note': 'No sunrise/sunset tomorrow (polar period)'
                        }

            return result

        except Exception as e:
            print(f"Error calculating solar data: {e}")
            return {
                'error': str(e),
                'current_altitude': 0.0,
                'current_azimuth': 0.0
            }

    def get_data(self):
        """Get fresh solar data, checking for config changes first"""
        with self.lock:
            self.check_config_changed()
            return self.calculate_solar_data()


def get_peer_credentials(sock):
    """Get peer credentials using SO_PEERCRED"""
    SO_PEERCRED = 17  # Linux constant
    creds = sock.getsockopt(socket.SOL_SOCKET, SO_PEERCRED, struct.calcsize('3i'))
    pid, uid, gid = struct.unpack('3i', creds)
    return pid, uid, gid


def handle_client(conn, cache):
    """Handle a client connection"""
    try:
        # Get peer credentials for security
        pid, uid, gid = get_peer_credentials(conn)

        # Verify it's from the same user
        if uid != os.getuid():
            print(f"Rejected connection from UID {uid} (expected {os.getuid()})")
            conn.close()
            return

        # Read request
        request = conn.recv(1024).decode().strip()

        if request == "GET":
            data = cache.get_data()
            response = json.dumps(data) + "\n"
            conn.sendall(response.encode())
        elif request == "RELOAD":
            cache.check_config_changed()
            conn.sendall(b"OK: Config reloaded\n")
        elif request == "STATUS":
            status = {
                'status': 'running',
                'config_file': str(cache.config_file),
                'config_exists': cache.config_file.exists(),
                'current_config': cache.config
            }
            response = json.dumps(status) + "\n"
            conn.sendall(response.encode())
        else:
            conn.sendall(b"ERROR: Unknown command (use GET, RELOAD, or STATUS)\n")

    except Exception as e:
        print(f"Error handling client: {e}")
        try:
            error_response = json.dumps({'error': str(e)}) + "\n"
            conn.sendall(error_response.encode())
        except:
            pass
    finally:
        conn.close()


def main():
    # Abstract socket name (starts with null byte)
    # Format: \0 + name (no filesystem path needed!)
    socket_name = f"\0argos-sunpos-{os.getuid()}"

    # Initialize cache
    print("Initializing solar position service...")
    print("Libraries loaded (pandas, pvlib) - ready for fast queries")
    cache = SolarDataCache()

    # Create Unix domain socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    # Bind to abstract namespace (no filesystem cleanup needed!)
    sock.bind(socket_name)
    sock.listen(5)

    print(f"Solar position service listening on abstract socket: {repr(socket_name)}")
    print("Service calculates fresh data on EVERY query for real-time accuracy")
    print("Config auto-reloads when argos-sunpos.json changes")
    print("")
    print("Irradiance modes:")
    print("  offline=true:  Ineichen model with Linke turbidity lookup (no internet, ±10-15% accuracy)")
    print("  offline=false: Simplified Solis with NASA POWER (±5-10% accuracy)")
    print("                 Location rounded to ~11km grid for privacy")

    # Accept connections
    try:
        while True:
            conn, _ = sock.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, cache), daemon=True)
            client_thread.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        sock.close()
        # No filesystem cleanup needed with abstract sockets!


if __name__ == '__main__':
    main()
