package main

type ModelInit struct {
	ObjectType string    `json:"docType"`
	Entity_Model_ID   string    `json:"Entity_Model_ID"` // the ID number of entity
	Param      []interface{} `json:"Param"`    // which domain the entity belongs to
	ModelAddress string      `json:"ModelAddress"`
	ModelType    string      `json:"ModelType"`
	TrainType    string       `json:"TrainType"`
}

type TrainStrategy struct {
	ObjectType string    `json:"docType"`
	Training_Round   string    `json:"Training_Round"` // the ID number of entity
	Reward_rules      []float64 `json:"Reward_rules"`    // which domain the entity belongs to
	Trainer_Account []string     `json:"Trainer_Account"`
}

type Parameter struct{
	Param_W [][]float64 `json:"Param_W"`
	Param_b []float64 `json:"Param_b"`
	Param_m []float64  `json:"Param_m"`
}
type Iot struct {
	ObjectType              string      `json:"docType"`
	AccountAddress          string      `json:"AccountAddress"`
	EntityID                string      `json:"EntityID"` // the ID number of entity
	Role                    string      `json:"Role"`
	Param                   []Parameter `json:"Param"`                   // which domain the entity belongs to
	ModelUpdateAddress_Info string      `json:"ModelUpdateAddress_Info"` // which domain the entity belongs to
	ModelUpdate_Hash        string      `json:"ModelUpdate_Hash"`
	//DomainName	string	`json:"DomainName"`		// which domain the entity belongs to
	//GenerateTime interface{}   //the time data generate
	//ObeserveData []byte   `json:"ObeserveData"`
	//Score        int     `json:"Score"`//the total price this account has
	WhoCanSee []string `json:"WhoCanSee"`
	//WhoIcanSee interface{} `json:"WhoIcanSee"`
	Price          float64    `json:"Price"` //the price of data
	AccountBalance float64   `json:"AccountBalance"`
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

type BlockAppend struct {
	BCindex string `json:"BCindex"`
	BCcontent Block `json:"BCcontent"`
	Winner string `json:"Winner"`
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

type HistoryItem struct {
	TxId	string
	Iot	    Iot
}

type Encrykeys struct {
	GenerateTime interface{}   //the time data generate
	Key []byte `json:"Key"`
	DurationTime interface{}
}
type ComputerRes struct{
	Total_grad_W [5][2]float64 `json:"Total_grad_W"`
	Total_grad_b [2]float64 `json:"Total_grad_b"`
	Total_cost [2]float64 `json:"Total_cost"`
}

type Consensus_committee struct{
	Name interface{} `json:"Name"`
	Members []Consensus_member `json:"Members"`
}

type Consensus_member struct{
	Name string `json:"Name"`
	Value float64 `json:"Value"`
}