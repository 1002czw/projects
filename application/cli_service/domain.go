/**
  @Author : hanxiaodong
*/
package cli_service

import (
	"fmt"
	"github.com/hyperledger/fabric-sdk-go/pkg/client/channel"
	"github.com/hyperledger/fabric-sdk-go/pkg/common/providers/fab"
	//"github.com/iot-data-sharing-project/bc-iot-net/consensus/blockGenerate"
	"time"
)

type ModelInit struct {
	ObjectType string        `json:"docType"`
	Entity_Model_ID   string    `json:"Entity_Model_ID"` // the ID number of entity
	Param      []interface{} `json:"Param"`    // which domain the entity belongs to
	ModelAddress string      `json:"ModelAddress"`
	ModelType    string      `json:"ModelType"`
	TrainType    string       `json:"TrainType"`
}
type Parameter struct {
	Param_W [][]float64 `json:"Param_W"`
	Param_b []float64   `json:"Param_b"`
	Param_m []float64   `json:"Param_m"`
}
type Iot struct {
	ObjectType              string      `json:"docType"`
	AccountAddress          string      `json:"AccountAddress"`
	EntityID                string      `json:"EntityID"` // the ID number of entity
	Role                    string      `json:"Role"`
	Param                   []Parameter `json:"Param"`                   // which domain the entity belongs to
	ModelUpdateAddress_Info string      `json:"ModelUpdateAddress_Info"` // which domain the entity belongs to
	ModelUpdate_Hash        string      `json:"ModelUpdate_Hash"`
	//GenerateTime interface{}   //the time data generate
	//ObeserveData []byte   `json:"ObeserveData"`
	//Score        int     `json:"Score"`//the total price this account has
	WhoCanSee []string `json:"WhoCanSee"`
	//WhoIcanSee interface{} `json:"WhoIcanSee"`
	Price          float64    `json:"Price"` //the price of data
	AccountBalance float64    `json:"AccountBalance"`
	Publickey      string `json:"Publickey"`

	DataSize          int `json:"Data_size"`
	AvailableResource float64 `json:"Available_Resource"`
	NetworkLatency    float64 `json:"Network_Latency"`
	RewardRule []Reward `json:"RewardRule"`

	Historys []HistoryItem // 当前iot的历史记录
}

type Reward struct{
	Name string `json:"Name"`
	Reward_value float64 `json:"Reward_Value"`
}
type Eval_consen_out struct{
	Name string `json:"name"`
	Judge_out bool `json:"judge_Out"`
	Time float64 `json:"time"`
	Sv_time int `json:"sv_time"`
	Contribution interface{}
}
type EvalCli struct{
	GroupName string  `json:"GroupName"`
	EvalValue []EvalCliDict  `json:"EvalValue"`
	Compute_SV_Round_times int `json:"Compute_SV_Round_times"`
	Winner string `json:"winner"`
	Winner_Eval_Results []EvalCliDict `json:"Winner_Eval_Results"`
}

type EvalCliDict struct{
	Name string `json:"Name"`
	Eval_value float64 `json:"Eval_value"`

}

type BlockAppend struct {
	BCindex string `json:"BCindex"`
	BCcontent Block `json:"BCcontent"`
	Winner string `json:"Winner"`
}

type Block struct       { //Block 是我们定义的结构体，它代表组成区块链的每一个块的数据模型
	Index int                //区块链中数据记录的位置
	Timestamp string    //时间戳，是自动确定的，并且是写入数据的时间
	Data string                  //假定我们现在做的是一个共享单车的区块链，Bike就是一定区域内的自行车数量
	Hash string             //是代表这个数据记录的SHA256标识符
	PrevHash string      //是链中上一条记录的SHA256标识符
	Difficulty int             //挖矿的难度
	Height int
	Nonce string           //PoW中符合条件的数字
	ConsensusType string
}
var Blockchain []Block // 存放区块数据
type Message struct{   //  定义结构体，请求的数据
	Data string
}

type Consensus_competing_result struct{
	Name string `json:"name"`
	Judge_out bool `json:"judge_Out"`
	Time time.Time `json:"time"`
	Nonce string `json:"nonce"`
	Shapley_result_addr string `json:"shapley_Result_Addr"`
	Aggre_result string `json:"aggre_Result"`
	Block Block `json:"block"`
}



type HistoryItem struct {
	TxId string
	Iot  Iot
}
type Encrykeys struct {
	GenerateTime interface{} //the time data generate
	Key          []byte      `json:"Key"`
	DurationTime interface{}
}

type ServiceSetup struct {
	ChaincodeID string
	Client      *channel.Client
}

type Consensus_committee struct{
	Name string `json:"Name"`
	Members []Consensus_member `json:"Members"`
}

type Consensus_member struct{
	Name interface{} `json:"Name"`
	Value float64 `json:"Value"`
}


func regitserEvent(client *channel.Client, chaincodeID, eventID string) (fab.Registration, <-chan *fab.CCEvent) {

	reg, notifier, err := client.RegisterChaincodeEvent(chaincodeID, eventID)
	if err != nil {
		fmt.Println("注册链码事件失败: %s", err)
	}
	return reg, notifier
}

func eventResult(notifier <-chan *fab.CCEvent, eventID string) error {
	select {
	case ccEvent := <-notifier:
		fmt.Printf("接收到链码事件: %v\n", ccEvent)
	case <-time.After(time.Second * 20):
		return fmt.Errorf("不能根据指定的事件ID接收到相应的链码事件(%s)", eventID)
	}
	return nil
}

