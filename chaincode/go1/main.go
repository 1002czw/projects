package main

import (
	"fmt"
	"github.com/hyperledger/fabric/bccsp"
	"github.com/hyperledger/fabric/bccsp/factory"
	"github.com/hyperledger/fabric/core/chaincode/shim"
	pb "github.com/hyperledger/fabric/protos/peer"
)

type Iotchaincode struct{
	bccspInst bccsp.BCCSP

}
//Initation the chaincode
func (t * Iotchaincode) Init(stub shim.ChaincodeStubInterface) pb.Response{
	return shim.Success(nil)
}

//Invoke function
func (t * Iotchaincode) Invoke(stub shim.ChaincodeStubInterface) pb.Response{
	// get function name and parameters first
	functionName , args := stub.GetFunctionAndParameters()
	/*tMap, err := stub.GetTransient()
	if err != nil {
		return shim.Error(fmt.Sprintf("Could not retrieve transient, err %s", err))
	}

	 */
	fmt.Println("Invoke is running: "+ functionName)
	//judge and choose the function
    if functionName == "registerInfo"{
		return t.registerInfo(stub,args)
    } else if functionName == "deleteInfo"{
		return t.deleteInfo(stub,args)
	} else if functionName == "queryInfo"{
		return t.queryInfo(stub,args)
	} else if functionName == "updateInfo"{
		return t.updateInfo(stub,args)
	} else if functionName == "queryInfoByDomain"{
		return t.queryInfoByDomain(stub,args)
	} else if functionName == "getHistorybyEntity"{
		return t.getHistorybyEntity(stub,args)
	} else if functionName == "getInfoByRange"{
		return t.getInfoByRange(stub,args)
	} else if functionName == "payMoneyforKey"{
		return t.payMoneyforKey(stub,args)
	} else if functionName == "submitKey"{
		return t.submitKey(stub,args)
	} else if functionName == "computerAvgModel"{
		return t.computerAvgModel(stub,args)
	} else {
		fmt.Println("Invoke did not find this func: "+functionName)
		return shim.Error("Received unknown function, please check your function name")
	}




}

//main function to run
func main(){
	factory.InitFactories(nil)
	err := shim.Start(&Iotchaincode{factory.GetDefault()})
	if err != nil{
		fmt.Printf("Error starting your Iotchaincode: %s", err)
	}
}