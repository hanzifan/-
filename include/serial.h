#ifndef SERIAL_H
#define SERIAL_H

#include <stdio.h>  
#include <string.h>  
#include <stdlib.h>  
  
#include <fcntl.h>  
#include <unistd.h>  
  
#include <termios.h> //set baud rate  
  
#include <sys/select.h>  
#include <sys/time.h>  
#include <sys/types.h>  
#include <errno.h>  
#include <sys/stat.h> 

class Serial{
public:
    Serial();
    ~Serial();
    int set_serial(int port);
    int serial_send(unsigned char* buffSenData, unsigned int sendDataNum);    //buffSenData should len 1024
    int serial_recieve(unsigned char* buffRcvData);

private:
    int openPort(int fd, int comport);
    int setOpt(int fd, int nSpeed, int nBits, char nEvent, int nStop);
    int readDataTty(int fd, unsigned char *rcv_buf, int TimeOut, int Len);
    int sendDataTty(int fd, unsigned char *send_buf, int Len);

private:
    int iSetOpt;//SetOpt 的增量i  
    int fdSerial; 
};

#endif