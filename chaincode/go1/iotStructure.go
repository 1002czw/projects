package main

type ModelInit struct {
	ObjectType string    `json:"docType"`
	EntityID   string    `json:"EntityID"` // the ID number of entity
	Param      []interface{} `json:"Param"`    // which domain the entity belongs to
}
type Parameter struct{
	Param_W [][]float64 `json:Param_W`
	Param_b []float64 `json:Param_b`
	Param_m []float64  `json:Param_m`
}
type Iot struct {
	ObjectType	string	`json:"docType"`
	EntityID	string	`json:"EntityID"` // the ID number of entity
	Param	[]Parameter	`json:"Param"`
	//DomainName	string	`json:"DomainName"`		// which domain the entity belongs to
    //GenerateTime interface{}   //the time data generate
    //ObeserveData []byte   `json:"ObeserveData"`
	//Score        int     `json:"Score"`//the total price this account has
	WhoCanSee []string `json:"WhoCanSee"`
	//WhoIcanSee interface{} `json:"WhoIcanSee"`
    Price   int     `json:"Price"`    //the price of data
    Account int `json:Account`
	Publickey string `json:Publickey`

	Historys	[]HistoryItem	// 当前iot的历史记录
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