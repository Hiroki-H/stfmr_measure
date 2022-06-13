#%%
import pyvisa as visa

rm = visa.ResourceManager()
list_insts = rm.list_resources()
print(list_insts)
#%%
# search instrument and query information
for index, inst in enumerate(list_insts):
    try:
        list_inst=rm.list_resources()
        access_inst = rm.open_resource(list_inst[index])
        inst_name=access_inst.query('*IDN?')
        print('GPIB {} is corresponding to {}'.format(list_inst[index],inst_name))
        access_inst.close()
    except:
        print(' Timeout expired before operation completed.')
        pass

#%%