package main

import (
	"fmt"

	"encoding/json"
	"io/ioutil"
	"os"
	//"time"
	"os/exec"
	"strconv"
	//"strings"
	"sync"
)

var wp sync.WaitGroup

type Iot1 struct {
	ObjectType      string `json:"docType"`
	AccountAddress  string `json:"AccountAddress"`
	EntityID        string `json:"EntityID"` // the ID number of entity
	Role            string `json:"Role"`
	ModelUpdateInfo string `json:"ModelUpdateInfo"` // which domain the entity belongs to
	//GenerateTime interface{}   //the time data generate
	//ObeserveData []byte   `json:"ObeserveData"`
	//Score        int     `json:"Score"`//the total price this account has
	WhoCanSee []string `json:"WhoCanSee"`
	//WhoIcanSee interface{} `json:"WhoIcanSee"`
	Price          int    `json:"Price"` //the price of data
	AccountBalance int    `json:"AccountBalance"`
	Publickey      string `json:"Publickey"`
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

func main() {
	winner_cli_filename := fmt.Sprintf("./eval_out_static_cli_1.json")
	winner_sv_data, err := ioutil.ReadFile(winner_cli_filename)
	if err != nil {
		fmt.Println("eval_out_static_cli file read err :", err)
	}
	var win_eval_client map[string]interface{}
	json.Unmarshal(winner_sv_data, &win_eval_client)
	Eval_client:=win_eval_client["Contributions"]
	var E EvalCli
	var e EvalCliDict
	for k, v := range Eval_client.(map[string]interface{}){
		e.Name = k
		fmt.Println(v.(map[string]interface{})["f1"])
		e.Eval_value,_ = strconv.ParseFloat(fmt.Sprintf("%.6f", v.(map[string]interface{})["f1"]), 64)
		E.EvalValue = append(E.EvalValue, e)
	}
	fmt.Println(E.EvalValue)
	/*
	v, _ := strconv.ParseFloat(fmt.Sprintf("%.3f", 0.2227), 64)
	fmt.Println(v)

	var f float64 = 0.35678
	fmt.Println(f)
	fmt.Printf("%.3f", f)

	s := "client_1"
	fmt.Println(len(s))
	index := (strings.Index(s, "_"))
	b := fmt.Sprintf("%c", s[index+1])
	fmt.Println(b)
	fmt.Printf("%T\n", b)

	cli_info, err := ioutil.ReadFile("./TEST.json")
	if err != nil {
		fmt.Printf("Client_%d read contri_statics error:%s\n", 1, err.Error())
	}
	var t EvalCli1
	t.GroupName = "AAA"
	var result map[string]interface{}
	json.Unmarshal(cli_info, &result)
	fmt.Println(result["C"])
	var tt EvalCliDict
	for k, v := range result {
		tt.Name = k
		tt.Value = v
		t.EvalValue = append(t.EvalValue, tt)
	}
	fmt.Println(t.EvalValue)
	fmt.Println(t.EvalValue[2])

	/*
		fmt.Println("###############################################################")
		fmt.Println("######## Clients download global mode && initializing #########")
		fmt.Println("###############################################################")
		//glo_model_path:="ShapleyValue"
		for i := 1; i <= 1; i++ {
			dp_mechanism:="Laplace"
			dp_clip:=20.0
			dp_epsilon:=20.0
			command :=fmt.Sprintf(`../application/users/client_%d/local_training.sh %s %f %f`,i,dp_mechanism,dp_clip,dp_epsilon)
			cmd1 := exec.Command("/bin/bash", "-c", command)
			output, err1 := cmd1.Output()
			if err1 != nil {
				fmt.Printf("Execute Shell:%s failed with error:%s", command, err1.Error())
				return
			}
			fmt.Printf("Execute Shell:%s finished :\n%s", command, string(output))
		}


		eval_client := make(map[string]float64)
		eval_client["A"]=0
		eval_client["B"]=0
		eval_client["C"]=0
		b,err :=json.Marshal(eval_client)
		eval_cli_filename := fmt.Sprintf("./TEST.json")
		err = ioutil.WriteFile(eval_cli_filename, b, 0777)
		if err != nil {
			fmt.Println("Write file error", err)
		}
		cc := EvalCli{
			GroupName: "SSS",
			EvalValue: b,
		}
		fmt.Println(eval_client)
		fmt.Println(b)
		fmt.Println(string(b))
		fmt.Println(cc)


		for{

			for i:=1;i<5;i++{

				go p(i)
				wp.Add(1)
			}
			index:=4
			go pp(index)

			wp.Wait()
			if index==4{
				break
			}
		}



		fmt.Println(cli_info)
		cli, err := json.Marshal(cli_info)
		fmt.Println(cli)*/

}
func p(i int) {
	defer wp.Done()
	a := i * i
	fmt.Println(a)
}
func pp(i int) {
	defer wp.Done()
	a := i * i
	fmt.Println(a)
}
func read() {
	command := fmt.Sprintf(`./read.sh`)
	cmd1 := exec.Command("/bin/bash", "-c", command)
	output, err1 := cmd1.Output()
	if err1 != nil {
		fmt.Printf("Execute Shell:%s failed with error:%s", command, err1.Error())
		return
	}
	fmt.Printf("Execute Shell:%s finished :\n%s", command, string(output))
}
func write() {
	command := fmt.Sprintf(`./write.sh`)
	cmd1 := exec.Command("/bin/bash", "-c", command)
	output, err1 := cmd1.Output()
	if err1 != nil {
		fmt.Printf("Execute Shell:%s failed with error:%s", command, err1.Error())
		return
	}
	fmt.Printf("Execute Shell:%s finished :\n%s", command, string(output))
}

func TestCompu(x, y int) {
	fmt.Printf("the sum is %d:", x+y)
}
func LocalTrainer_Init(command1, command2 string) {
	cmd1 := exec.Command("/bin/bash", "-c", command1)
	output, err1 := cmd1.Output()
	if err1 != nil {
		fmt.Printf("Execute Shell:%s failed with error:%s", command1, err1.Error())
		return
	}
	fmt.Printf("Execute Shell:%s finished :\n%s", command1, string(output))
	cmd2 := exec.Command("/bin/bash", "-c", command2)
	output, err2 := cmd2.Output()
	if err2 != nil {
		fmt.Printf("Execute Shell:%s failed with error:%s", command2, err2.Error())
		return
	}
	fmt.Printf("Execute Shell:%s finished with output:\n%s", command2, string(output))

}
func AsyncFunc(index int) {
	sum := 0
	for i := 0; i < 10000; i++ {
		sum += 1
	}
	fmt.Printf("线程%d, sum为:%d\n", index, sum)
}

func LocalInit(i int) {
	command := fmt.Sprintf(`./test.sh %d`, i)

	cmd := exec.Command("/bin/bash", "-c", command)
	output, err := cmd.Output()
	if err != nil {
		fmt.Printf("Execute Shell:%s failed with error:%s", command, err.Error())
		return
	}
	fmt.Printf("Execute Shell:%s finished with output:\n%s", command, string(output))

}
func HasDir(path string) (bool, error) {
	_, _err := os.Stat(path)
	if _err == nil {
		return true, nil
	}
	if os.IsNotExist(_err) {
		return false, nil
	}
	return false, _err
}

//创建文件夹
func CreateDir(path string, i int) {
	_exist, _err := HasDir(path)
	if _err != nil {
		fmt.Printf("get Dir path error -> %v\n", _err)
		return
	}
	if _exist {
		fmt.Printf("client-%d has already exist, and now delete it and recreate it------>", i)
		RemoveDir(path)
		err := os.MkdirAll(path, os.ModePerm)
		if err != nil {
			fmt.Printf("Create dir error: -> %v\n", err)
		} else {
			fmt.Printf("Create client-%d direction successful!\n", i)
			fmt.Println("######################################################################")
		}
	} else {
		err := os.MkdirAll(path, os.ModePerm)
		if err != nil {
			fmt.Printf("Create dir error: -> %v\n", err)
		} else {
			fmt.Printf("Create client-%d direction successful!\n", i)
			fmt.Println("######################################################################")
		}
	}
}

//删除文件
func RemoveFile(path string) error {
	_err := os.Remove(path)
	return _err
}

//删除文件夹
func RemoveDir(path string) error {
	_err := os.RemoveAll(path)
	return _err
}
