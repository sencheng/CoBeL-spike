// This an example program for reading a JSON file using the library JSONCPP in C++
/* 
   **Here, the JSON file "network_params_spikingnet.json" is read and a specific value 
   from a particular key (as an example "action:orientation_sel_dic:mov_step")is read. 

   **To install the JSONCPP library in linux use the following command-
   sudo apt-get install libjsoncpp-dev

   **For more details about the library and other installation ways, check the following website-
   https://en.wikibooks.org/wiki/JsonCpp

   **To run the program in terminal use -ljsoncpp
   For example: g++ readJSON.cpp -o readJSON -ljsoncpp
   then run by: ./readJSON
*/

#include <iostream>
#include <fstream>
#include <jsoncpp/json/json.h>  //Header file for JSONCPP library
using namespace std;

int main() {
	ifstream ifs("network_params_spikingnet.json");
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);
    //The following line shows how to access the keys in the JSON file.
    cout<<"required value is "<<obj["action"]["orientation_sel_dic"]["mov_step"].asFloat()<<endl;
    ifs.close();
}