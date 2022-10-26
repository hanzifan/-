#include <iostream>
#include "serial.h"

Serial::Serial(){
    int iSetOpt = 0;
    int fdSerial = 0;
}

Serial::~Serial(){
}

int Serial::openPort(int fd, int comport){
    if (comport == 1)  
    {  
        fd = open("/dev/ttyTHS1", O_RDWR | O_NOCTTY | O_NDELAY);  
        if (-1 == fd)  
        {  
            perror("Can't Open Serial Port");  
            return(-1);  
        }  
        else  
        {  
            printf("open ttyTHS1 .....\n");  
        }  
    }  
    else if (comport == 2)  
    {  
        fd = open("/dev/ttyS2", O_RDWR | O_NOCTTY | O_NDELAY);  
        if (-1 == fd)  
        {  
            perror("Can't Open Serial Port");  
            return(-1);  
        }  
        else  
        {  
            printf("open ttyS2 .....\n");  
        }  
    } 
  
    if (fcntl(fd, F_SETFL, 0)<0)  
    {  
        printf("fcntl failed!\n");  
    }  
    else  
    {  
        printf("fcntl=%d\n", fcntl(fd, F_SETFL, 0));  
    }  
    if (isatty(STDIN_FILENO) == 0)  
    {  
        printf("standard input is not a terminal device\n");  
    }  
    else  
    {  
        printf("is a tty success!\n");  
    }  
    printf("fd-open=%d\n", fd);  
    return fd;
}

int Serial::setOpt(int fd, int nSpeed, int nBits, char nEvent, int nStop)  
{  
    struct termios newtio, oldtio;  
    if (tcgetattr(fd, &oldtio) != 0)  
    {  
        perror("SetupSerial 1");  
        return -1;  
    }  
    bzero(&newtio, sizeof(newtio));  
    newtio.c_cflag |= CLOCAL | CREAD;  
    newtio.c_cflag &= ~CSIZE;  
  
    switch (nBits)  
    {  
    case 7:  
        newtio.c_cflag |= CS7;  
        break;  
    case 8:  
        newtio.c_cflag |= CS8;  
        break;  
    }  
  
    switch (nEvent)  
    {  
    case 'O':                     //奇校验  
        newtio.c_cflag |= PARENB;  
        newtio.c_cflag |= PARODD;  
        newtio.c_iflag |= (INPCK | ISTRIP);  
        break;  
    case 'E':                     //偶校验  
        newtio.c_iflag |= (INPCK | ISTRIP);  
        newtio.c_cflag |= PARENB;  
        newtio.c_cflag &= ~PARODD;  
        break;  
    case 'N':                    //无校验  
        newtio.c_cflag &= ~PARENB;  
        break;  
    }  
  
    switch (nSpeed)  
    {  
    case 2400:  
        cfsetispeed(&newtio, B2400);  
        cfsetospeed(&newtio, B2400);  
        break;  
    case 4800:  
        cfsetispeed(&newtio, B4800);  
        cfsetospeed(&newtio, B4800);  
        break;  
    case 9600:  
        cfsetispeed(&newtio, B9600);  
        cfsetospeed(&newtio, B9600);  
        break;  
    case 115200:  
        cfsetispeed(&newtio, B115200);  
        cfsetospeed(&newtio, B115200);  
        break;  
    default:  
        cfsetispeed(&newtio, B9600);  
        cfsetospeed(&newtio, B9600);  
        break;  
    }  
    if (nStop == 1)  
    {  
        newtio.c_cflag &= ~CSTOPB;  
    }  
    else if (nStop == 2)  
    {  
        newtio.c_cflag |= CSTOPB;  
    }  
    newtio.c_cc[VTIME] = 0;  
    newtio.c_cc[VMIN] = 0;  
    tcflush(fd, TCIFLUSH);  
    if ((tcsetattr(fd, TCSANOW, &newtio)) != 0)  
    {  
        perror("com set error");  
        return -1;  
    }  
    printf("set done!\n");  
    return 0;  
}

int Serial::readDataTty(int fd, unsigned char *rcv_buf, int TimeOut, int Len)  
{  
    int retval;  
    fd_set rfds;  
    struct timeval tv;  
    int ret, pos;  
    tv.tv_sec = TimeOut / 1000;  //set the rcv wait time    
    tv.tv_usec = TimeOut % 1000 * 1000;  //100000us = 0.1s    
  
    pos = 0;  
    while (1)  
    {  
        FD_ZERO(&rfds);  
        FD_SET(fd, &rfds);  
        retval = select(fd + 1, &rfds, NULL, NULL, &tv);  
        if (retval == -1)  
        {  
            perror("select()");  
            break;  
        }  
        else if (retval)  
        {  
            ret = read(fd, rcv_buf + pos, 1);  
            if (-1 == ret)  
            {  
                break;  
            }  
  
            pos++;  
            if (Len <= pos)  
            {  
                break;  
            }  
        }  
        else  
        {  
            break;  
        }  
    }  
  
    return pos;  
}  
  
int Serial::sendDataTty(int fd, unsigned char *send_buf, int Len)  
{  
    ssize_t ret;  
  
    ret = write(fd, send_buf, Len);  
    if (ret == -1)  
    {  
        printf("write device error\n");  
        return -1;  
    }  
  
    return 1;  
}

int Serial::set_serial(int port){
    //openPort  
    if ((fdSerial = openPort(fdSerial, port))<0)//1--"/dev/ttyS0",2--"/dev/ttyS1",3--"/dev/ttyS2",4--"/dev/ttyUSB0" 小电脑上是2--"/dev/ttyS1"  
    {  
        perror("open_port error");  
        return -1;  
    }  

    if ((iSetOpt = setOpt(fdSerial, 9600, 8, 'N', 1))<0)  
    {  
        perror("set_opt error");  
        return -1;  
    }  
    printf("Serial fdSerial=%d\n", fdSerial);  
  
    tcflush(fdSerial, TCIOFLUSH);//清掉串口缓存  
    fcntl(fdSerial, F_SETFL, 0);
}

int Serial::serial_send(unsigned char* buffSenData, unsigned int sendDataNum){
    sendDataTty(fdSerial, buffSenData, sendDataNum); 
}

int Serial::serial_recieve(unsigned char* buffRcvData){ 
    readDataTty(fdSerial, buffRcvData, 2, 1024);
    // std::cout <<  int(buffRcvData[0]) << " " <<  int(buffRcvData[1]) << " "  <<  int(buffRcvData[2]) << " " <<  int(buffRcvData[3]) << " " 
    //         <<  int(buffRcvData[4]) << " " <<  int(buffRcvData[5]) << " " <<  int(buffRcvData[6]) << std::endl; 
}