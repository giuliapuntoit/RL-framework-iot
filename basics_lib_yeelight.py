from yeelight import discover_bulbs
from time import sleep

discover_bulbs()

from yeelight import Bulb
bulb = Bulb("192.168.1.183")

# Turn the bulb on.
bulb.turn_on()

sleep(4)
# Turn the bulb off.
bulb.turn_off()

sleep(4)
# Toggle power.
bulb.toggle()
