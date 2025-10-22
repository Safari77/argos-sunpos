#!/usr/bin/env python3
import socket
import json
import os
import struct
import sys

def categorize_uv_index(uv_value):
    """Categorize UV index value according to WHO standards"""
    if uv_value < 3:
        return "Low 🟢"
    elif uv_value < 6:
        return "Moderate 🟡"
    elif uv_value < 8:
        return "High 🟠"
    elif uv_value < 11:
        return "Very High 🔴"
    else:
        return "Extreme 🟣"

def get_peer_credentials(sock):
    """Get peer credentials using SO_PEERCRED"""
    SO_PEERCRED = 17
    creds = sock.getsockopt(socket.SOL_SOCKET, SO_PEERCRED, struct.calcsize('3i'))
    pid, uid, gid = struct.unpack('3i', creds)
    return pid, uid, gid

def query_solar_service():
    """Query the solar position service via abstract Unix socket"""
    socket_name = f"\0argos-sunpos-{os.getuid()}"

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(socket_name)

        # Verify server UID
        try:
            pid, uid, gid = get_peer_credentials(sock)
            if uid != os.getuid():
                print(f"Security error: Server UID {uid} != our UID {os.getuid()}", file=sys.stderr)
                sock.close()
                return None
        except Exception as e:
            print(f"Failed to verify server credentials: {e}", file=sys.stderr)
            sock.close()
            return None

        sock.sendall(b"GET\n")

        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
            if b'\n' in chunk:
                break

        sock.close()
        data = json.loads(response.decode().strip())
        return data

    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        return None

# Query the service
data = query_solar_service()

if data:
    # Main display
    print(f"☀️ Alt: {data['current_altitude']:.2f}° Az: {data['current_azimuth']:.2f}°")
    print("---")

    # Sunrise
    if data['sunrise']:
        print(f"Sunrise: {data['sunrise']['time']} (Az: {data['sunrise']['azimuth']:.2f}°)")
    elif 'sunrise_note' in data:
        print(f"Sunrise: {data['sunrise_note']}")
    else:
        print("Sunrise: None")

    # Solar noon
    if data['solar_noon']:
        print(f"Max Altitude: {data['solar_noon']['max_altitude']:.2f}° at {data['solar_noon']['time']}")
    else:
        print("Max Altitude: N/A")

    # Sunset
    if data['sunset']:
        print(f"Sunset: {data['sunset']['time']} (Az: {data['sunset']['azimuth']:.2f}°)")
    elif 'sunset_note' in data:
        print(f"Sunset: {data['sunset_note']}")
    else:
        print("Sunset: None")

    # Solar irradiance
    if 'irradiance' in data:
        print("---")
        irr = data['irradiance']

        if 'error' not in irr:
            # Show mode indicator
            mode_indicator = "ONLINE" if not irr.get('offline_mode', True) else "OFFLINE"
            print(f"📊 Solar Irradiance ({mode_indicator})")

            # Display irradiance values in W/m²
            print(f"GHI: {irr['ghi']:.0f} W/m²")
            print(f"DNI: {irr['dni']:.0f} W/m²")
            print(f"DHI: {irr['dhi']:.0f} W/m²")

            # Show airmass if available and valid (useful indicator of atmospheric path)
            airmass = irr.get('airmass')
            if airmass is not None and airmass == airmass:  # Check for NaN (NaN != NaN)
                print(f"Airmass: {airmass:.2f}")

            # Show atmospheric conditions if available (online mode only)
            if 'atmospheric' in irr:
                atm = irr['atmospheric']
                uv_str = f"{atm['uv_index_max']:.1f}" if atm.get('uv_index_max') is not None else "N/A"
                print(f"Conditions: {atm['temperature']:.1f}°C, {atm['humidity']:.0f}% RH, AOD={atm['aod_550nm']:.3f}, UV={uv_str}")

            # Show note if present (e.g., nighttime)
            if 'note' in irr:
                print(f"({irr['note']})")

            # Current UV Index
            if 'uv_index_current' in irr:
                uv = irr['uv_index_current']
                print(f"UV Index (now): {uv:.1f} ({categorize_uv_index(uv)})")

            # UV Index at zenith (maximum for today)
            if irr.get('uv_index_zenith') is not None:
                uv_zenith = irr['uv_index_zenith']
                print(f"UV Index (max): {uv_zenith:.1f} ({categorize_uv_index(uv_zenith)})")

            # NASA POWER UV Index (recent day maximum)
            if 'atmospheric' in irr and irr['atmospheric'].get('uv_index_max') is not None:
                uv = irr['atmospheric']['uv_index_max']
                print(f"UV Index (NASA, max): {uv:.1f}")
        else:
            print(f"Irradiance: Error - {irr['error']}")

    # Day type indicator with daylight info
    if 'day_type' in data:
        day_types = {
            'normal': '☀️ Normal day',
            'short_day': '🌄 Short day',
            'polar_day': '☀️ Polar day (sun does not set)',
            'polar_night_civil_twilight': '🌆 Polar night (civil twilight)',
            'polar_night_nautical_twilight': '🌃 Polar night (nautical twilight)',
            'polar_night_astronomical_twilight': '🌌 Polar night (astronomical twilight)',
            'polar_night_total': '🌑 Polar night (total darkness)'
        }
        print("---")
        print(day_types.get(data['day_type'], data['day_type']))

        # Show daylight duration and polar end dates
        if data['day_type'] == 'polar_day' and 'daylight_duration' in data:
            print(f"Daylight: {data['daylight_duration']['formatted']}")

        if 'polar_ends' in data:
            if 'formatted' in data['polar_ends']:
                print(f"{data['polar_ends']['formatted']} ({data['polar_ends']['date']})")
            elif 'note' in data['polar_ends']:
                print(f"{data['polar_ends']['note']}")
        elif data['day_type'] in ['normal', 'short_day'] and 'daylight_duration' in data:
            # Handle both formatted and note cases
            if 'formatted' in data['daylight_duration']:
                print(f"Daylight: {data['daylight_duration']['formatted']}")
            elif 'note' in data['daylight_duration']:
                print(f"Daylight: {data['daylight_duration']['note']}")

            # Show tomorrow's change
            if 'daylight_change' in data:
                if 'formatted' in data['daylight_change']:
                    print(f"Tomorrow: {data['daylight_change']['formatted']}")
                elif 'note' in data['daylight_change']:
                    print(f"Tomorrow: {data['daylight_change']['note']}")

    # Config info
    if 'config' in data:
        print("---")
        print(f"Location: lat={data['config']['latitude']:.2f}°, lon={data['config']['longitude']:.2f}°, alt={data['config']['altitude']:.2f} m")
else:
    print("☀️ --")
    print("---")
    print("Service unavailable")
    print("Start: systemctl --user start argos-sunpos | bash='systemctl --user start argos-sunpos' terminal=false")
