# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import array
import binascii
import serial
import serial.tools.list_ports

class UART():
    def __init__(self, 
                port_name="/dev/ttyUSB0", 
                baud_rate=9600,
                time_out=5):
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.time_out = time_out

    def available_port(self):
        port_list = list(serial.tools.list_ports.comports())
        if len(port_list) <= 0:
            print("Not available port")
        else:
            port_list_0 = list(port_list[0])
            serial_name = port_list_0[0]
            serial_find = serial.Serial(serial_name, 9600, timeout=60)
            print("Available port: ", serial_find)

    def open_serial(self):
        self.ser = serial.Serial(self.port_name, self.baud_rate, timeout=self.time_out)
    
    def serial_state(self):
        print("Serial port name: {}".format(self.ser.name))
        print("Serial read timeout: {}s".format(self.ser.timeout))
    
    def send_msg(self, msg, mode=1):
        """
        mode=1 -> 10
        mode=2 -> hex, for example: [0xaa, 0x12]
        """
        print("Sending: {}".format(msg))
        if mode == 1:
            self.ser.write(msg)
        if mode == 2:
            msg = array.array("B", msg).tostring()
            self.ser.write(msg)        

    def receive_msg(self, buffer=2, mode=1, hex_mode=False):
        """
        mode=1 -> read buffer
        mode=2 -> read lines
        hex_mode=True -> return hex message
        """
        if mode == 1:
            msg = self.ser.read(buffer)
        if mode == 2:
            msg = self.ser.readline()
        if hex_mode:
            msg = binascii.hexlify(msg).decode("utf-8")

        print("Receive: {}".format(msg))
        return msg

    def close_serial(self):
        self.ser.close()

if __name__ == '__main__':
    uart = UART(port_name="/dev/ttyUSB0", 
                baud_rate=9600,
                time_out=1)
    
    uart.open_serial()
    uart.serial_state()
    while True:
        # uart.send_msg([0xaa, 0x12], mode=2)
        uart.receive_msg(buffer=2, mode=1, hex_mode=True)

    uart.close_serial()
    