from bledom import BleLedDevice
from bledom.device import Effects
from bleak import BleakScanner, BleakClient
import logging

TARGET_NAME = "BLEDOM0B"

class LedController:
    def __init__(self):
        self.client = None
        self.led = None

    async def connect(self):
        logging.debug("Scanning LEDs...")
        devices = await BleakScanner.discover()

        target = None
        for device in devices:
            if device.name and TARGET_NAME in device.name:
                target = device
                break

        if not target:
                raise RuntimeError(f"Device '{TARGET_NAME}' not found.")

        logging.debug(f"Connecting to {target.name}...")
        self.client = BleakClient(target)
        await self.client.connect()

        self.led = await BleLedDevice.new(self.client)
        logging.debug("Connected.")

    async def white(self):
        if self.led:
            logging.debug("Solid white")
            await self.led.set_color(255, 255, 255)

    async def blink_white(self):
        if self.led:
            logging.debug("Blinking white")
            await self.led.set_effect(Effects.BLINK_WHITE)
            await self.led.set_effect_speed(50)

    async def rainbow(self):
        if self.led:
            logging.debug("Rainbow")
            await self.led.set_effect(Effects.JUMP_RED_GREEN_BLUE_YELLOW_CYAN_MAGENTA_WHITE)

    async def disconnect(self):
        if self.client:
            logging.debug("Disconnecting...")
            await self.client.disconnect()
            logging.debug("Disconnected.")
