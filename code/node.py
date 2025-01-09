class Node():
    def __init__(self, id, x, y, is_data_node=False, data_packets=0, storage_capacity=0):
        self.id = id    #node id 
        self.x = x      #x-coordinate
        self.y = y      #y-coordinate
        self.is_data_node = is_data_node    #either data node or data sink (ME) 
        self.data_packets = data_packets    #number of data packets
        self.storage_capacity = storage_capacity    #maximum capacity of data storage 
    
    # set node location by coordinates 
    def setLocation(self, x, y):
        self.x = x
        self.y = y
    
    # return node location: Array[x-coordinate, y-coordinate]
    def getLocation(self):
        return [self.x, self.y]
    
    # return node's x-coordinate: Double 
    def get_x(self):
        return self.x
    
    # return node's y-coordinate: Double 
    def get_y(self):
        return self.y
    
    # return node type. either data node or data sink: Boolean 
    def is_DN(self):
        if self.is_data_node:
            return True
        else:
            return False
    
    # return the number of data package: Int 
    def get_data_packets(self):
        return self.data_packets

    # return the maximum capacity of data storage: Int 
    def get_storage_capacity(self):
        return self.storage_capacity
    
    # return every attribure: Tuple(node_id, x-coordinate, y-coordinate, is_data_node, data_packets, storage_capacity)
    def get_info(self):
        return (self.id, self.x, self.y, self.is_data_node, self.data_packets, self.storage_capacity)

    # return the id of a node: Int
    def get_id(self):
        return self.id
    
    # print node_id and number of data_packets for corresponding node
    def list_data_packet(self):
        print("ID: {}, Number of data packet: {}\n".format(self.id, self.data_packets))