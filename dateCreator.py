# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 14:22:51 2021

@author: SMedepalli
"""
#######program to increment dates

file=open("datesForProphet.txt", "a")
day=10
month=10
year=2000
hr=0
mins=0
sec=1
for i in range (1000):
    
    
    
    if(mins>=10):
      if(sec==60):
            mins+=1
            #file.write("mins inc")
            sec=0
            if(mins>=10):
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+"0"+str(hr)+":"+str(mins)+":0"+str(sec))
                file.write("\n")
            else:
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+"0"+str(hr)+":"+str(mins)+":0"+str(sec))
                file.write("\n")
      else:
            if(sec>=10):
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+"0"+str(hr)+":"+str(mins)+":"+str(sec))
                file.write("\n")
            else:
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+"0"+str(hr)+":"+str(mins)+":0"+str(sec))
                file.write("\n")
    else:
      if(sec==60):
            mins+=1
            #file.write("mins inc")
            sec=0
            if(mins>=10):
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+"0"+str(hr)+":"+str(mins)+":0"+str(sec))
                file.write("\n")
            else:
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+"0"+str(hr)+":0"+str(mins)+":0"+str(sec))
                file.write("\n")
      else:
            if(sec>=10):
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+"0"+str(hr)+":0"+str(mins)+":"+str(sec))
                file.write("\n")
            else:
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+"0"+str(hr)+":0"+str(mins)+":0"+str(sec))
                file.write("\n")
    sec+=1    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    if(hr>=10):
        if(mins>=10):
            file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+str(hr)+":"+str(mins)+":"+sec)
            file.write("\n")
            mins+=1
        else:
            file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+"0"+str(hr)+":0"+str(mins)+":"+sec)
            file.write("\n")
            mins+=1
            
    elif(mins==60):
        hr+=1
        mins=0
        if(hr>=10):
            if(mins>=10):
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+str(hr)+":"+str(mins)+":"+sec)
                file.write("\n")
                mins+=1
            else:
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+str(hr)+":0"+str(mins)+":"+sec)
                file.write("\n")
                mins+=1
        else:
            if(mins>=10):
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" 0"+str(hr)+":"+str(mins)+":"+sec)
                file.write("\n")
                mins+=1
            else:
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" 0"+str(hr)+":0"+str(mins)+":"+sec)
                file.write("\n")
                mins+=1
            
            
                
    
        
    else:
        if(mins>=10):
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+str(hr)+":"+str(mins)+":"+sec)
                file.write("\n")
                mins+=1
        else:
                file.write(str(year)+"-"+str(month)+"-"+str(day)+" "+"0"+str(hr)+":0"+str(mins)+":"+sec)
                file.write("\n")
                mins+=1
    
    """
    
    
file.close()
print(year)
