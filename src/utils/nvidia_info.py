'''
Author      : 
Contact     :
LastEditors: Please set LastEditors
LastEditTime: 2020-11-28 05:48:04
Description : 
'''
import pynvml 

def get_free_device_ids():
    pynvml.nvmlInit()
    num_device = pynvml.nvmlDeviceGetCount()
    # print(num_device)
    free_device_id = []
    for i in range(num_device):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        men_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(men_info.total,men_info.free)
        # import pdb; pdb.set_trace()
        # print(men_info.free)
        # print(men_info.total)
        if men_info.free >= men_info.total*0.90:
            free_device_id.append(i)
    return free_device_id


if __name__ == "__main__":
    print("Available:",get_free_device_ids())

    import pdb; pdb.set_trace()