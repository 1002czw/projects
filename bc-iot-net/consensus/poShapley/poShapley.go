package poShapley

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/iot-data-sharing-project/application/cli_service"
	"github.com/iot-data-sharing-project/bc-iot-net/consensus/blockGenerate"
	"io/ioutil"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

var wp sync.WaitGroup

const (
	glo_model_path      = "ShapleyValue"
	consensus_FL_metric = "f1"
)

func PoShapley(round, cli_num int, serviceSetup1 cli_service.ServiceSetup, transactions [][]byte) cli_service.Consensus_competing_result {
	c_times := 1
	winner := ""
	for {
		judge_Chan := make(chan cli_service.Eval_consen_out, cli_num)
		for i := 1; i <= cli_num; i++ {
			go Aggre_Eval_Consensus(judge_Chan, i, round, c_times)
			wp.Add(1)
		}
		wp.Wait()
		fmt.Println("go finished")
		winner = Aggre_consensus_judge(judge_Chan, round, c_times, serviceSetup1)
		if winner != "" {
			break
		}
		c_times += 1
	}
	tree := blockGenerate.NewMerkleTree(transactions)
	Merkleroot_hash := hex.EncodeToString(tree.RootNode.Data)
	var newBlock cli_service.Block //新区块
	t := time.Now()
	oldBlock := cli_service.Blockchain[len(cli_service.Blockchain)-1]
	newBlock.Index = oldBlock.Index + 1 //区块的增加，index也加一
	newBlock.Timestamp = t.String()     //时间戳
	newBlock.Data = Merkleroot_hash
	newBlock.PrevHash = oldBlock.Hash //新区块的PrevHash存储上一个区块的Hash
	newBlock.Height = oldBlock.Height + 1
	newBlock.Difficulty = 0
	newBlock.Hash = CalculateHash(newBlock)
	newBlock.ConsensusType = "poShapley"
	inputInfo := cli_service.Consensus_competing_result{
		Name:                winner,
		Judge_out:           true,
		Time:                time.Now(),
		Shapley_result_addr: "",
		Block:               newBlock,
	}
	return inputInfo
}

func Aggre_Eval_Consensus(judgeConsensusChan chan cli_service.Eval_consen_out, i, round, times int) {
	defer wp.Done()
	fmt.Printf("Client_%d Aggre_Eval_Consensus at round %d starting......\n", i, round)
	command := fmt.Sprintf(`~/go/src/github.com/iot-data-sharing-project/application/users/client_%d/aggre_Consensus.sh %d %d`, i, i, times)
	cmd1 := exec.Command("/bin/bash", "-c", command)
	output, err1 := cmd1.Output()
	if err1 != nil {
		fmt.Printf("Client_%d Aggre_Eval_Consensus at round %d times %dth--->: failed with error:%s\n", i, round, times, err1.Error())
		return
	}
	fmt.Printf("Client_%d Aggre_Eval_Contribution at round %d times %dth --->: finished :\n%s", i, round, times, string(output))
	contri_statics_filename := fmt.Sprintf("/home/czw/go/src/github.com/iot-data-sharing-project/application/users/global_Model/global_Model_Server/%s/eval_out/eval_out_static_cli_%d.json", glo_model_path, i)
	cli_info, err := ioutil.ReadFile(contri_statics_filename)
	if err != nil {
		fmt.Printf("Client_%d read contri_statics error:%s\n", i, err.Error())
	}
	var Eval_result cli_service.Eval_consen_out
	json.Unmarshal(cli_info, &Eval_result)
	judgeConsensusChan <- Eval_result

}

func Aggre_consensus_judge(judgeConsen_Chan chan cli_service.Eval_consen_out, round, times int, serviceSetup1 cli_service.ServiceSetup) (winner string) {
	//judge
	close(judgeConsen_Chan)
	result_judgeTrue_dic := make(map[float64]string)
	for val := range judgeConsen_Chan {
		result_judge := val
		if result_judge.Judge_out == true {
			result_judgeTrue_dic[result_judge.Time] = result_judge.Name
		}
	}
	if len(result_judgeTrue_dic) != 0 {
		s := make([]float64, 0, len(result_judgeTrue_dic))
		for key := range result_judgeTrue_dic {
			s = append(s, key)
		}
		sort.Float64s(s)
		winner := result_judgeTrue_dic[s[0]]
		index := (strings.Index(winner, "_"))
		b := fmt.Sprintf("%c", winner[index+1])
		winner_cli_filename := fmt.Sprintf("/home/czw/go/src/github.com/iot-data-sharing-project/application/users/global_Model/global_Model_Server/%s/eval_out/eval_out_static_cli_%s.json", glo_model_path, b)
		winner_sv_data, err := ioutil.ReadFile(winner_cli_filename)
		if err != nil {
			fmt.Println("eval_out_static_cli file read err :", err)
		}
		var win_eval_client map[string]interface{}
		json.Unmarshal(winner_sv_data, &win_eval_client)
		Eval_client := win_eval_client["Contributions"]
		var E cli_service.EvalCli
		var e cli_service.EvalCliDict
		for k, v := range Eval_client.(map[string]interface{}) {
			e.Name = k
			e.Eval_value, _ = strconv.ParseFloat(fmt.Sprintf("%.6f", v.(map[string]interface{})[consensus_FL_metric]), 64)
			E.Winner_Eval_Results = append(E.Winner_Eval_Results, e)
		}
		affi := fmt.Sprintf("Round_%d_Contribution_Eval", round)
		E.GroupName = affi + glo_model_path
		E.Compute_SV_Round_times = times
		E.Winner = winner

		command := "~/go/src/github.com/iot-data-sharing-project/application/users/utils/judge_Consensus.sh "
		cmd1 := exec.Command("/bin/bash", "-c", command)
		_, err1 := cmd1.Output()
		if err1 != nil {
			fmt.Printf("Update average of SV value at round %d times %dth--->: failed with error:%s\n", round, times, err1.Error())
			
		}
		last_eval_cli_filename := fmt.Sprintf("/home/czw/go/src/github.com/iot-data-sharing-project/application/users/global_Model/global_Model_Server/%s/cliEval.json", glo_model_path)
		sv_data, err := ioutil.ReadFile(last_eval_cli_filename)
		if err != nil {
			fmt.Println("err :", err)
		}
		var eval_client map[string]float64
		json.Unmarshal(sv_data, &eval_client)
		var e1 cli_service.EvalCliDict
		for k, v := range eval_client {
			e1.Name = k
			e1.Eval_value, _ = strconv.ParseFloat(fmt.Sprintf("%.6f", v), 64)
			E.EvalValue = append(E.EvalValue, e1)
		}
		Eval_msg, err := serviceSetup1.UploadContriInfoIOT(E)
		if err != nil {
			fmt.Println("Consensus winner Update error")
			fmt.Println(err.Error())
		} else {
			fmt.Println()
			fmt.Printf("--->Consensus winner sucess, blockchain tx: %s \n", Eval_msg)
			fmt.Println("************************************************************************************************")
		}

		return winner

	} else {
		//Update
		fmt.Printf("Judge_Consensus  at round %d times %dth--->: finished, Update the average of SV and start the nest round:\n%s", round, times)
		command := "~/go/src/github.com/iot-data-sharing-project/application/users/utils/judge_Consensus.sh "
		cmd1 := exec.Command("/bin/bash", "-c", command)
		_, err1 := cmd1.Output()
		if err1 != nil {
			fmt.Printf("Judge_Consensus at round %d times %dth--->: failed with error:%s\n", round, times, err1.Error())
			return
		}
		eval_cli_filename := fmt.Sprintf("/home/czw/go/src/github.com/iot-data-sharing-project/application/users/global_Model/global_Model_Server/%s/cliEval.json", glo_model_path)
		sv_data, err := ioutil.ReadFile(eval_cli_filename)
		if err != nil {
			fmt.Println("err :", err)
			return
		}
		var eval_client map[string]float64
		json.Unmarshal(sv_data, &eval_client)
		var E cli_service.EvalCli
		affi := fmt.Sprintf("Round_%d_Contribution_Eval", round)
		E.GroupName = affi + glo_model_path
		E.Compute_SV_Round_times = times
		E.Winner = ""
		var e cli_service.EvalCliDict
		for k, v := range eval_client {
			e.Name = k
			e.Eval_value, _ = strconv.ParseFloat(fmt.Sprintf("%.6f", v), 64)
			E.EvalValue = append(E.EvalValue, e)
		}

		Eval_msg, err := serviceSetup1.UploadContriInfoIOT(E)
		if err != nil {
			fmt.Println("Clients contribution Update error")
			fmt.Println(err.Error())
		} else {
			fmt.Println()
			fmt.Printf("--->Clients contribution Update sucess, blockchain tx: %s \n", Eval_msg)
			fmt.Println("************************************************************************************************")
		}

		return ""
	}

}

func CalculateHash(block cli_service.Block) string {
	record := strconv.Itoa(block.Index) + block.Timestamp + block.Data + block.PrevHash + block.Nonce
	h := sha256.New()
	h.Write([]byte(record))
	hashed := h.Sum(nil)
	return hex.EncodeToString(hashed)
}
