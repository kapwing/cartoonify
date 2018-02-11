import importlib
import logging


class Gpio:
    """
    interface to raspi GPIO
    """

    def __init__(self):
        self._capture_pin = 4
        self._status_pin = 2
        self._logger = logging.getLogger(self.__class__.__name__)
        self.gpio = None
        try:
            self.gpio = importlib.import_module('RPi.GPIO')
        except ImportError as e:
            self._logger.exception(e)
            print('raspi gpio module not found, continuing...')

    def setup(self, capture_callback):
        """setup GPIO pin to trigger callback function when capture pin goes low

        :return:
        """
        self.gpio.setmode(self.gpio.BCM)
        self.gpio.setup(self._capture_pin, self.gpio.IN, pull_up_down=self.gpio.PUD_UP)
        self.gpio.add_event_detect(self._capture_pin, self.gpio.FALLING, callback=capture_callback, bouncetime=200)
        self.gpio.setup(self._status_pin, self.gpio.OUT)
        self.gpio.output(self._status_pin, False)

    def status_pin(self, state):
        """set status pin high/low

        :param bool state:
        :return:
        """
        if self.available():
            self.gpio.output(self._status_pin, state)

    def available(self):
        """return true if gpio package is available

        :return:
        """
        return self.gpio is not None

    def close(self):
        if self.available():
            self.gpio.cleanup()