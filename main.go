package main

import (
	"bufio"
	_ "encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/iot-data-sharing-project/application/cli_sdkInIt"
	"github.com/iot-data-sharing-project/application/cli_service"
	"github.com/iot-data-sharing-project/bc-iot-net/consensus/blockGenerate"
	"github.com/iot-data-sharing-project/bc-iot-net/consensus/poAggre"
	"github.com/iot-data-sharing-project/bc-iot-net/consensus/poShapley"
	"github.com/iot-data-sharing-project/bc-iot-net/consensus/pow"
	"github.com/iot-data-sharing-project/bc-iot-net/consensus/pos"
	"github.com/iot-data-sharing-project/bc-iot-net/consensus/dPoS"
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"runtime"
	"encoding/csv"
)

var wp sync.WaitGroup

var transactions_Init []string

const (
	configFile1 = "./application/cli_sdkInIt/config.yaml"
	configFile2 = "./application/cli_sdkInIt/config2.yaml"
	initialized = false
	deskey      = "0123abcd"
	pub         = `-----BEGIN RSA PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCzSHe6Hdc3Vv4iazAd8k5kqH2J
v4u5MPvwDHRxej8Fp5wZE1BtuuoATgc1EIQHZ3W9lwfaWvuO7iXDcKyP4+pXgmjo
LPEzKEePDS9e3FWehr1r4AmhXksrbFhj2L69yY0qw0rHdPAk+VJxfcflSPa5FHu6
biNezqngByoHp097YwIDAQAB
-----END RSA PUBLIC KEY-----`
	pri = `-----BEGIN RSA PRIVATE KEY-----
MIICWwIBAAKBgQCzSHe6Hdc3Vv4iazAd8k5kqH2Jv4u5MPvwDHRxej8Fp5wZE1Bt
uuoATgc1EIQHZ3W9lwfaWvuO7iXDcKyP4+pXgmjoLPEzKEePDS9e3FWehr1r4Amh
XksrbFhj2L69yY0qw0rHdPAk+VJxfcflSPa5FHu6biNezqngByoHp097YwIDAQAB
AoGAbRcdcyyRNmVCwiLC5pX4SZuUt+rL1GYQByMs/5fJHWG9xCxSdlKT7JeOHyXr
kK3NoQ1zg9R02aXjaKb4+Z1Pi+a8uuaaoiTUxd0STrsza0pegZ53zytl/+wKvBUv
sRmjaMUvNbGt12CJrPob6QTWaQO7M+3c+3EgMuO00YLEJzECQQDlxSybvpnltp/P
LA6XibFmbs9wKvH1iM0hXtJITOUMZWDZQZrjKCQ5Jwf82SnEwrZ6BoAaQBTgQgei
yyU+qdVfAkEAx7/aTvNZIcGXKqaAIuMswiiomi22aWGAUP4Cmce9IEvQErxxD4ft
wlUnxL5IeQYfvAfq2/oiVakUuVD/WdU0fQJAWgMWaKUQkScsD/MIfGEauDAs35pT
U4cWJT6KTnyhNmv4tuO2k8rD9gfOX0WL0WMeLUuin6X/B6OWbjX7D0NlLQJAI9S6
aGcmmfOMFk5/tcJiiQGaHO4ORqECz4SBGuzFdEGlNXcwIMUVVULJO3WWsn7yulwB
OSyJdCw8V3G8HHUuUQJAJRTbOfkJHG1MwV7YutSeZpkwSY4Sz/rYOtEYKoH1f+Te
/LAcyhTmOX7wCQvKA0khDxtm7fg5ASOQcZQPNzUI/g==
-----END RSA PRIVATE KEY-----`
	IotCC               = "Fe_test202307_13"
	glo_model_path      = "ShapleyValue"
	Epoch               = 20
	consensus_FL_metric = "f1"
	consensusType_first_stage       = "dpos"
	consensusType_second_stage       = "dpos"
	consensusType_third_stage       = "dpos"
	Committee_Size      = 10
	train_on_going      = true
	train_off_going      = false
)

func main() {

	initInfo := &cli_sdkInIt.InitInfo{

		ChannelID: "iotchannel",
		OrgAdmin:  "Admin",
		OrgName:   "Org1",
		UserName:  "User1",
	}

	sdk, err := cli_sdkInIt.SetupSDK(configFile1, initialized)
	if err != nil {
		fmt.Printf(err.Error())
		return
	}
	defer sdk.Close()

	channelClient, err := cli_sdkInIt.NewChannelCli(sdk, initInfo)
	if err != nil {
		fmt.Println(err.Error())
		return
	}
	fmt.Println(channelClient)
	serviceSetup1 := cli_service.ServiceSetup{
		ChaincodeID: IotCC,
		Client:      channelClient,
	}

	genesisBlock := cli_service.Block{
		Index:         0,
		Timestamp:     time.Now().String(),
		Data:          "This is Genesis Block",
		Hash:          "",
		PrevHash:      "00000000000000000000",
		Difficulty:    0,
		Height:        0,
		Nonce:         "0",
		ConsensusType: "Genesis Block",
	}
	genesisBlock.Hash=blockGenerate.CalculateHash(genesisBlock)

	cli_service.Blockchain = append(cli_service.Blockchain, genesisBlock)

	append_genesis_block := cli_service.BlockAppend{
		BCindex:   "BC_0_Genesis_Block",
		BCcontent: genesisBlock,
		Winner:    "System generate block",
	}

	genesis_block_generate_msg, err_genesis_block_generate_msg := serviceSetup1.AppendBlockIOT(append_genesis_block)
	transactions_Init = append(transactions_Init, genesis_block_generate_msg)
	if err_genesis_block_generate_msg != nil {
		fmt.Println("block append error")
		fmt.Println(err_genesis_block_generate_msg.Error())
	} else {
		fmt.Printf("Genesis block generate sucess, blockchain tx: %s \n", genesis_block_generate_msg)
	}

	//实体链上信息注册
	fmt.Println()
	fmt.Println("========================================== IoT Data Sharing Requester Register ==========================================")
	fmt.Println()
	req_filename := fmt.Sprintf("./application/users/requester_%d/reqInfo.json", 1)
	req_info, err_req_info := ioutil.ReadFile(req_filename)
	if err_req_info != nil {
		fmt.Println("read req_info is err", err_req_info)
	}
	fmt.Println()
	fmt.Println("************************************  Define the size of collaborative training group  ************************************ ")

	input := bufio.NewScanner(os.Stdin)
	input.Scan()
	cli_num, err := strconv.Atoi(input.Text())
	if err != nil {
		fmt.Println("the number of clients strconv is err", err)
	}
	//读取的数据为json格式，需要进行解码
	RewardRule_init := cli_service.Iot{}
	err = json.Unmarshal(req_info, &RewardRule_init)
	//define the reward rules

	for i := 1; i <= cli_num; i++{
		var RewardRule_entity cli_service.Reward
		entity_key:=fmt.Sprintf("entity_%d",i)
		RewardRule_entity.Name=entity_key
		RewardRule_entity.Reward_value=5
		RewardRule_init.RewardRule=append(RewardRule_init.RewardRule,RewardRule_entity)
	}
	requester_register_msg, err := serviceSetup1.RegisterIOT(RewardRule_init)
	if err != nil {
		fmt.Println("entity register error")
		fmt.Println(err.Error())
	} else {
		fmt.Printf("requester %d register sucess, blockchain tx: %s \n", 1, requester_register_msg)
	}
	transactions_Init = append(transactions_Init, requester_register_msg)
	fmt.Println()
	fmt.Println("=============================  IoT Data Sharing Providers Register  =============================")
	fmt.Println()
	eval_client := make(map[string]float64)
	for i := 1; i <= cli_num; i++ {

		dirpath := fmt.Sprintf("./application/users/client_%d", i)
		CreateDir(dirpath, i)
		cli_data, err_read_cli_data := ioutil.ReadFile("./application/users/utils/cliInfo.json")
		if err_read_cli_data != nil {
			fmt.Println("err 有", err_read_cli_data)
			return
		}
		cli_filename := fmt.Sprintf("./application/users/client_%d/cliInfo.json", i)
		err_cli := ioutil.WriteFile(cli_filename, cli_data, 0777)
		if err_cli != nil {
			fmt.Println("Write file error", err_cli)
		}
		cli_info, err_cliInfo_read := ioutil.ReadFile(cli_filename)
		if err_cliInfo_read != nil {
			fmt.Println("Read the new file error", err_cliInfo_read)
		}
		//读取的数据为json格式，需要进行解码
		entity_cli_info := cli_service.Iot{}
		err = json.Unmarshal(cli_info, &entity_cli_info)
		change := fmt.Sprintf("Prov_IoT00%d", i)//
		entity_cli_info.AccountAddress = change
		entity_cli_info.EntityID = change
		entity_cli_info.Publickey = change
		save_entity, err_save_entity := json.Marshal(entity_cli_info)
		if err_save_entity != nil {
			fmt.Println("save_entity Marshal error", err_save_entity)
		}
		err_cli_write := ioutil.WriteFile(cli_filename, save_entity, 0777)
		if err_cli_write != nil {
			fmt.Println("Write the new file error", err_cli_write)
		}
		cliInfo_register_msg, err_register_msg := serviceSetup1.RegisterIOT(entity_cli_info)
		transactions_Init = append(transactions_Init, cliInfo_register_msg)
		if err_register_msg != nil {
			fmt.Println("entity register error")
			fmt.Println(err_register_msg.Error())
		} else {
			fmt.Printf("provider client %d register sucess, blockchain tx: %s \n", i, cliInfo_register_msg)
			fmt.Println("************************************************************************************************")
		}
		client_name := fmt.Sprintf("client_%d", i)
		eval_client[client_name] = 0
	}

	fmt.Println()
	fmt.Println("==================================== Global training model initialization ====================================")
	fmt.Println()
	command := "./application/users/requester_1/run_init_globalModel.sh"
	cmd := exec.Command("/bin/bash", "-c", command)
	output, err := cmd.Output()
	if err != nil {
		fmt.Printf("Execute Shell:%s failed with error:%s", command, err.Error())
		return
	}
	fmt.Printf("Execute Shell:%s finished with output:\n%s", command, string(output))
	taskPath := fmt.Sprintf("./application/users/requester_%d/modelInit.json", 1)
	initModel, err := ioutil.ReadFile(taskPath)
	if err != nil {
		fmt.Println("read file error= %v", err)
	}
	//读取的数据为json格式，需要进行解码
	dataTask := cli_service.ModelInit{}
	err = json.Unmarshal(initModel, &dataTask)
	taskReleaseMsg, err := serviceSetup1.ReleaseTaskIOT(dataTask)
	transactions_Init = append(transactions_Init, taskReleaseMsg)
	if err != nil {
		fmt.Println(err.Error())
	} else {
		fmt.Println("Global model initialization and release success, blockchain tx: " + taskReleaseMsg)
	}

	b, err := json.Marshal(eval_client)
	eval_cli_filename := fmt.Sprintf("./application/users/global_Model/global_Model_Server/%s/cliEval.json", glo_model_path)
	err = ioutil.WriteFile(eval_cli_filename, b, 0777)
	if err != nil {
		fmt.Println("Write file error", err)
	}
	var E cli_service.EvalCli
	E.GroupName = glo_model_path + "Contribution_Eval"
	E.Compute_SV_Round_times = 0
	E.Winner = ""
	var e cli_service.EvalCliDict
	for k, v := range eval_client {
		e.Name = k
		e.Eval_value = v
		E.EvalValue = append(E.EvalValue, e)
	}
	Init_eval_msg, err := serviceSetup1.UploadContriInfoIOT(E)
	transactions_Init = append(transactions_Init, Init_eval_msg)
	if err != nil {
		fmt.Println("Clients contribution Initialization error")
		fmt.Println(err.Error())
	} else {
		fmt.Println()
		fmt.Printf("--->Clients contribution Initialization sucess, blockchain tx: %s \n", Init_eval_msg)
		fmt.Println("************************************************************************************************")
	}
	Init_Start_Time := time.Now()
	fmt.Println()
	fmt.Println("================================ The first phase--Initialization-- block generation by DPoS ================================")
	fmt.Println()
	// generate consensus committee
	cli_committee, err_read_cli_committee := ioutil.ReadFile("./application/users/utils/basic_model/client_fraction.json")
	if err_read_cli_committee != nil {
		fmt.Println("generate consensus committee readfile err 有", err_read_cli_committee)
		return
	}
	var cli_committee_Map map[string]interface{}
	err = json.Unmarshal(cli_committee, &cli_committee_Map)
	if err != nil {
		fmt.Printf("cli_committee Json to Map error:%s\n", err)
		return
	}
	sort_cli_committee := sortMapByValue2(cli_committee_Map)

	var consensus_committee PairList
	if len(sort_cli_committee) <Committee_Size{
		consensus_committee = sort_cli_committee
	}else {
		consensus_committee = sort_cli_committee[0:Committee_Size]
	}
	var dpos_committee cli_service.Consensus_committee
	var dpos_committee_member cli_service.Consensus_member
	dpos_committee.Name = "DPoS_Committee"
	for k, _ := range (consensus_committee) {
		//读取的数据为json格式，需要进行解码
		dpos_committee_member.Name = consensus_committee[k].Key
		dpos_committee_member.Value = consensus_committee[k].Value
		dpos_committee.Members = append(dpos_committee.Members, dpos_committee_member)
	}

	consensus_committee_msg, err := serviceSetup1.UploadConsensusCommitteeIOT(dpos_committee)
	if err != nil {
		fmt.Println("Clients consensus_committee Initialization error")
		fmt.Println(err.Error())
	} else {
		fmt.Println()
		fmt.Printf("--->Clients consensus_committee Initialization sucess, blockchain tx: %s \n", consensus_committee_msg)
		fmt.Println("************************************************************************************************")
	}

	winner_Init := getWinnerbyConsensus(consensusType_first_stage, 0, cli_num, serviceSetup1, transactions_Init, train_off_going)
	fmt.Printf("New block generate by %s is valid\n", winner_Init.Name)
	fmt.Printf("============= New block apeending, Winner done the data sharing initlizating Tx packing operation   ===============\n")
	cli_service.Blockchain = append(cli_service.Blockchain, winner_Init.Block)
	append_generate_block_index0:=fmt.Sprintf("%d_BCFL-Datasharing-Init_Block",0)
	append_generate_block0 := cli_service.BlockAppend{
		BCindex:   append_generate_block_index0,
		BCcontent: winner_Init.Block,
		Winner:    winner_Init.Name,
	}

	generate_block_generate_msg0, err_generate_block_generate_msg0 := serviceSetup1.AppendBlockIOT(append_generate_block0)
	if err_generate_block_generate_msg0 !=nil{
		fmt.Printf("append_generate_block is error:%s",err_generate_block_generate_msg0)
	}else{
		fmt.Printf("append_generate_block is successful, the transaction is:%s",generate_block_generate_msg0)
	}
	elapsed := time.Since(Init_Start_Time)
	seconds := elapsed.Milliseconds()
	seconds2Str := strconv.FormatInt(seconds,10)
	file_experiment, err := os.OpenFile("experiment.csv",os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		fmt.Println("Failed to create CSV file:", err)
		return
	}


	// 创建 CSV writer
	writer_experiment := csv.NewWriter(file_experiment)

	headers := []string{"Initialize Phase", "BCFL-training Phase", "Reward-updating Phase", "Type-of-Consensus", "Nun-of-Group"}
	if err := writer_experiment.Write(headers); err != nil {
		fmt.Println("Failed to write headers:", err)
		return
	}
	data := [][]string{
		{seconds2Str, "0", "0", consensusType_first_stage, fmt.Sprintf("%s",cli_num)},
		}
	for _, rowData := range data {
		if err := writer_experiment.Write(rowData); err != nil {
			fmt.Println("Failed to write data:", err)
			return
		}
	}
	writer_experiment.Flush()
	file_experiment.Close()


	fmt.Println()
	fmt.Println("================================ Clients download global model && initializing ================================")
	fmt.Println()
	runtime.GOMAXPROCS(runtime.NumCPU())
	for i := 1; i <= cli_num; i++ {
		//command1 :=fmt.Sprintf(`./application/users/utils/copyBasicModel.sh %d`,i)
		//command2 :=fmt.Sprintf(`./application/users/client_%d/init_Client.sh %d %s`,i,i,glo_model_path)

		go LocalTrainer_Init(i, glo_model_path)
		wp.Add(1)
		//time.Sleep(60*time.Second)

	}
	wp.Wait()
	fmt.Println("===================================  Clients initializing finished  ===================================")
	fmt.Println()
	var time_static []int64
	for round := 1; round <= Epoch; round++ {
		train_Start_Time := time.Now()
		bt, errt := json.Marshal(eval_client)
		eval_cli_filename_bt := fmt.Sprintf("./application/users/global_Model/global_Model_Server/%s/cliEval.json", glo_model_path)
		err = ioutil.WriteFile(eval_cli_filename_bt, bt, 0777)
		if err != nil {
			fmt.Println("Write file error", errt)
		}

		var transactions_Training []string

		if round == 1{
			reward_update := cli_service.Iot{}
			err = json.Unmarshal(req_info, &reward_update)
			//define the reward rules

			for i := 1; i <= cli_num; i++{
				var RewardRule_entity cli_service.Reward

				entity_key:=fmt.Sprintf("Prov_IoT00%d",i)
				RewardRule_entity.Name=entity_key
				RewardRule_entity.Reward_value=5
				reward_update.RewardRule=append(reward_update.RewardRule,RewardRule_entity)

			}
			_, reward_update_err := serviceSetup1.UpdateIOT(reward_update)
			if reward_update_err != nil {
				fmt.Println("entity reward_init error")
				fmt.Println(err.Error())
			} else {

			}
		}

		fmt.Println()
		fmt.Printf("====== Data Providers decide training strategy and training local model at Round %d, and Upload it to Blockchain ======\n", round)
		fmt.Println()
		TIME := make(map[interface{}]float64)
		//runtime.GOMAXPROCS(runtime.NumCPU())
		for loop := 1; loop <= cli_num/5; loop++ {

			for i := loop *5-4; i <= loop *5; i++ {

				go LocalTrainer_Train(i, round, serviceSetup1, TIME)
				wp.Add(1)
			}
			wp.Wait()
		}
		sortTime := sortMapByValue(TIME)

		for k, _ := range sortTime {
			//读取的数据为json格式，需要进行解码
			index := sortTime[k].Key
			account := fmt.Sprintf("Prov_IoT00%d", index)
			Model_IPFS_address := fmt.Sprintf("www.client_%d/round %d/local_training_model.com", index, round)
			Model_HASH_Verify := fmt.Sprintf("Client_%d/round %d/local_training_model_hash", index, round)
			entity_Update := cli_service.Iot{
				AccountAddress:          account,
				ModelUpdateAddress_Info: Model_IPFS_address,
				ModelUpdate_Hash:        Model_HASH_Verify,
			}
			fmt.Println()
			fmt.Println("Upload to blockchain operations  -----> ")
			register_update_msg, err_register_update_msg := serviceSetup1.UpdateIOT(entity_Update)
			transactions_Training = append(transactions_Training, register_update_msg)
			if err_register_update_msg != nil {
				fmt.Println("Model training info Update error")
				fmt.Println(err.Error())
			} else {
				command_simulate := fmt.Sprintf("./application/users/utils/simulate_Upload2BC.sh %d %s", index, glo_model_path)
				cmd := exec.Command("/bin/bash", "-c", command_simulate)
				_, err_simu := cmd.Output()
				if err_simu != nil {
					fmt.Printf("Execute Shell:%s failed with error:%s", command, err_simu.Error())
					return
				}
				fmt.Printf("Client_%d Upload local training model at round %d sucess, blockchain tx: %s \n", index, round, register_update_msg)
				fmt.Println("************************************************************************************************")

			}

		}
		fmt.Println()
		fmt.Printf("=============================  Consensus-based Aggregation Process at Round %d   =============================\n", round)
		fmt.Println()
		check_result := false
		var winner cli_service.Consensus_competing_result

		t_strat:=time.Now().Unix()
		for {

			if check_result {
				fmt.Printf("New block generate by %s is valid\n", winner.Name)
				fmt.Printf("============= New block apeend, Winner  done    the    Global Model   Aggregation   operation   ===============\n")
				cli_service.Blockchain = append(cli_service.Blockchain, winner.Block)
				append_generate_block_index:=fmt.Sprintf("%d_BCFL_Train_Block",round)
				append_generate_block := cli_service.BlockAppend{
					BCindex:   append_generate_block_index,
					BCcontent: winner.Block,
					Winner:    winner.Name,
				}

				generate_block_generate_msg, err_generate_block_generate_msg := serviceSetup1.AppendBlockIOT(append_generate_block)
				if err_generate_block_generate_msg !=nil{
					fmt.Printf("append_generate_block is error:%s",err_generate_block_generate_msg)
				}
				transactions_Training=append(transactions_Training,generate_block_generate_msg)

				cli_win := strings.ToLower(winner.Name)
				command_Aggregation := fmt.Sprintf("./application/users/%s/aggregation_Global.sh %s ", cli_win, cli_win)
				cmd_command_Aggregation := exec.Command("/bin/bash", "-c", command_Aggregation)
				_, err_simu := cmd_command_Aggregation.Output()
				if err_simu != nil {
					fmt.Printf("Execute Shell:%s failed with error:%s \n", command_Aggregation, err_simu.Error())
					return
				}
				fmt.Printf("%s Global Model Aggregation at round %d sucess \n", winner, round)
				fmt.Println("************************************************************************************************")
				break

			} else {
				if round == 1 {
					Transactions_collection := append(transactions_Init, transactions_Training...)
					winner = getWinnerbyConsensus(consensusType_second_stage, round, cli_num, serviceSetup1, Transactions_collection, train_on_going)
				} else {
					winner = getWinnerbyConsensus(consensusType_second_stage, round, cli_num, serviceSetup1, transactions_Training, train_on_going)
				}

				fmt.Println(winner)
				t_end:=time.Now().Unix()
				time_static=append(time_static,t_end-t_strat)
				fmt.Printf("%s based consensus at round %d sucess \n", consensusType_second_stage, round)
				fmt.Printf("============================= Winner   is     %s   =============================\n", winner.Name)
				fmt.Printf("=============  Transmit block and others check the  new block   ===============\n")

				checkChannel := make(chan bool, cli_num-1)
				oldBlock := cli_service.Blockchain[len(cli_service.Blockchain)-1]
				check_result = blockGenerate.ChechBlock(cli_num, winner, oldBlock, checkChannel)
			}
		}

		elapsed_train := time.Since(train_Start_Time)
		seconds_train := elapsed_train.Milliseconds()
		seconds_train2Str := strconv.FormatInt(seconds_train,10)

		ReUp_Start_Time:= time.Now()
		fmt.Println()
		fmt.Printf("============================= Rewarding for Training Round %d and Update reward strategy at Round %d  =============================\n", round,round+1)
		fmt.Println()
		var transactions_payment []string
		for i := 1; i <= cli_num; i++{
			From_key:=fmt.Sprintf("Req_IoT00%d",1)
			To_key:=fmt.Sprintf("Prov_IoT00%d",i)
			reward_payment_msg, reward_payment_err := serviceSetup1.PayMoneyforKeyIOT(From_key,To_key)
			if reward_payment_err != nil {
				fmt.Println("entity reward_payment error")
				fmt.Println(err.Error())
			} else {
				fmt.Printf("requester %d pay for Prov_IoT00_%d sucess, blockchain tx: %s \n", 1,i, reward_payment_msg)
			}
			transactions_payment=append(transactions_payment,reward_payment_msg)

		}

		reward_update2 := cli_service.Iot{}
		err = json.Unmarshal(req_info, &reward_update2)
		//define the reward rules
		for i := 1; i <= cli_num; i++{
			var RewardRule_entity cli_service.Reward

			entity_key:=fmt.Sprintf("Prov_IoT00%d",i)
			RewardRule_entity.Name=entity_key
			RewardRule_entity.Reward_value=3.5
			reward_update2.RewardRule=append(reward_update2.RewardRule,RewardRule_entity)

		}
		reward_update2_msg, reward_update2_err := serviceSetup1.UpdateIOT(reward_update2)
		if reward_update2_err != nil {
			fmt.Println("entity reward_update error")
			fmt.Println(err.Error())
		} else {
			fmt.Println("entity reward_update success")
		}
		transactions_payment=append(transactions_payment,reward_update2_msg)
		fmt.Println()
		fmt.Printf("============================= Global Model Update For Next Training Round %d   =============================\n", round+1)
		fmt.Println()
		for i := 1; i <= cli_num; i++ {
			update_command := fmt.Sprintf(`./application/users/client_%d/update_GlobalModel.sh %d`, i, i)
			cmd1 := exec.Command("/bin/bash", "-c", update_command)
			output_Update_globalmodel, err1 := cmd1.Output()
			if err1 != nil {
				fmt.Printf("Client_%d Global Model Update For Next Training Round:%s failed with error:%s\n", i, update_command, err1.Error())
				return
			}
			fmt.Printf("Client_%d Global Model Update For Next Training Round:%s finished :%s\n", i, update_command, string(output_Update_globalmodel))

		}

		fmt.Println()
		fmt.Println("================================ The third phase --Rewarding && Updating-- block generation by DPoS ================================")
		fmt.Println()

		// update consensus committee

		sort_cli_committee_update := sortMapByValue2(cli_committee_Map)

		var consensus_committee_update PairList
		if len(sort_cli_committee_update) <Committee_Size{
			consensus_committee_update = sort_cli_committee_update
		}else {
			consensus_committee_update = sort_cli_committee_update[0:Committee_Size]
		}
		var dpos_committee_update cli_service.Consensus_committee
		var dpos_committee_member_update cli_service.Consensus_member
		dpos_committee_update.Name = "DPoS_Committee"
		for k, _ := range (consensus_committee_update) {
			//读取的数据为json格式，需要进行解码
			dpos_committee_member_update.Name = consensus_committee_update[k].Key
			dpos_committee_member_update.Value = consensus_committee_update[k].Value
			dpos_committee_update.Members = append(dpos_committee_update.Members, dpos_committee_member_update)
		}

		consensus_committee_msg_update, err := serviceSetup1.UploadConsensusCommitteeIOT(dpos_committee_update)
		if err != nil {
			fmt.Println("Clients consensus_committee Update error")
			fmt.Println(err.Error())
		} else {
			fmt.Println()
			fmt.Printf("--->Clients consensus_committee Update sucess, blockchain tx: %s \n", consensus_committee_msg_update)
			fmt.Println("************************************************************************************************")
		}
		winner_Re_UP := getWinnerbyConsensus(consensusType_third_stage, 0, cli_num, serviceSetup1, transactions_payment,train_off_going)
		fmt.Printf("New block generate by %s is valid\n", winner_Re_UP.Name)
		fmt.Printf("============= New block apeend, Winner done data sharing Reward-Updating Tx packing operation   ===============\n")
		cli_service.Blockchain = append(cli_service.Blockchain, winner_Re_UP.Block)
		append_generate_block_index2:=fmt.Sprintf("%d_BCFL-Datasharing-Reward-Update_Block",round)
		append_generate_block2 := cli_service.BlockAppend{
			BCindex:   append_generate_block_index2,
			BCcontent: winner_Init.Block,
			Winner:    winner_Init.Name,
		}

		generate_block_generate_msg2, err_generate_block_generate_msg2 := serviceSetup1.AppendBlockIOT(append_generate_block2)
		if err_generate_block_generate_msg0 !=nil{
			fmt.Printf("append_generate_block is error:%s",err_generate_block_generate_msg2)
		}else{
			fmt.Printf("append_generate_block is successful, the transaction is:%s",generate_block_generate_msg2)
		}

		elapsed_ReUp := time.Since(ReUp_Start_Time)
		seconds_ReUp := elapsed_ReUp.Milliseconds()
		seconds_ReUp2Str := strconv.FormatInt(seconds_ReUp,10)

		file_experiment1, err := os.OpenFile("experiment.csv",os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
		if err != nil {
			fmt.Println("Failed to create CSV file:", err)
			return
		}


		// 创建 CSV writer
		writer_experiment1 := csv.NewWriter(file_experiment1)

		data1 := [][]string{
			{"0", seconds_train2Str, seconds_ReUp2Str, consensusType_second_stage+"+"+consensusType_third_stage, fmt.Sprintf("%s",cli_num)},
		}
		for _, rowData := range data1 {
			if err := writer_experiment1.Write(rowData); err != nil {
				fmt.Println("Failed to write data:", err)
				return
			}
		}
		writer_experiment1.Flush()
		file_experiment1.Close()

	}
	fmt.Println(time_static)


}

func select_Clients(a, b int) []int {
	var client []int
	rad := rand.New(rand.NewSource(time.Now().UnixNano()))
	//rand.Seed(time.Now().UnixNano())
	for len(client) < a {
		num := rad.Intn(b)
		//查重
		exist := false
		for _, v := range client {
			if v == num {
				exist = true
				break
			}
		}
		if !exist {
			client = append(client, num)
		}

	}

	return client

}

func appendSave2txt(file string, data []string) {

	// 以追加模式打开文件，当文件不存在时生成文件
	f, err := os.OpenFile(file, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0666)
	defer f.Close()
	if err != nil {
		panic(err)
	}

	// 写入文件
	for i := 0; i < len(data); i++ {
		f.Write([]byte(data[i]))
		f.WriteString(" ")
		// 当 n != len(b) 时，返回非零错误
	}
	f.WriteString("\n")
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
		}
	} else {
		err := os.MkdirAll(path, os.ModePerm)
		if err != nil {
			fmt.Printf("Create dir error: -> %v\n", err)
		} else {
			fmt.Printf("Create client-%d direction successful!\n", i)
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

func LocalTrainer_Init(i int, glo_model_path string) {
	defer wp.Done()
	command1 := fmt.Sprintf(`./application/users/utils/copyBasicModel.sh %d`, i)
	command2 := fmt.Sprintf(`./application/users/client_%d/init_Client.sh %d %s`, i, i, glo_model_path)
	cmd1 := exec.Command("/bin/bash", "-c", command1)
	output, err1 := cmd1.Output()
	if err1 != nil {
		fmt.Printf("Execute Shell:%s failed with error:%s\n", command1, err1.Error())
		return
	}
	fmt.Printf("Execute Shell:%s finished :%s\n", command1, string(output))
	cmd2 := exec.Command("/bin/bash", "-c", command2)
	output, err2 := cmd2.Output()
	if err2 != nil {
		fmt.Printf("Execute Shell:%s failed with error:%s\n", command2, err2.Error())
		return
	}
	fmt.Printf("Execute Shell:%s finished with output:%s\n", command2, string(output))

}

func LocalTrainer_Train(i, round int, serviceSetup1 cli_service.ServiceSetup, TIME map[interface{}]float64) (RETURN_TIME map[interface{}]float64) {
	defer wp.Done()
	startTime := time.Now().UnixNano()
	dp_mechanism := "no_dp" //"Laplace"
	dp_clip := 20.0
	dp_epsilon := 20.0
	command := fmt.Sprintf(`./application/users/client_%d/local_training.sh %d %s %f %f`, i, i, dp_mechanism, dp_clip, dp_epsilon)
	cmd1 := exec.Command("/bin/bash", "-c", command)
	output, err1 := cmd1.Output()
	if err1 != nil {
		fmt.Printf("Client_%d local training at round %d --->: failed with error:%s\n", i, round, err1.Error())
		return
	}
	fmt.Printf("Client_%d local training at round %d --->: finished :\n%s", i, round, string(output))
	endTime := time.Now().UnixNano()
	TIME[i] = float64(endTime-startTime) / 1e9
	return TIME

}

func Aggre_Eval_Consensus(judgeConsensusChan chan cli_service.Eval_consen_out, i, round, times int) {
	defer wp.Done()
	fmt.Printf("Client_%d Aggre_Eval_Consensus at round %d starting......\n", i, round)
	command := fmt.Sprintf(`./application/users/client_%d/aggre_Consensus.sh %d %d`, i, i, times)
	cmd1 := exec.Command("/bin/bash", "-c", command)
	output, err1 := cmd1.Output()
	if err1 != nil {
		fmt.Printf("Client_%d Aggre_Eval_Consensus at round %d times %dth--->: failed with error:%s\n", i, round, times, err1.Error())
		return
	}
	fmt.Printf("Client_%d Aggre_Eval_Contribution at round %d times %dth --->: finished :\n%s", i, round, times, string(output))
	contri_statics_filename := fmt.Sprintf("./application/users/global_Model/global_Model_Server/%s/eval_out/eval_out_static_cli_%d.json", glo_model_path, i)
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
		winner_cli_filename := fmt.Sprintf("./application/users/global_Model/global_Model_Server/%s/eval_out/eval_out_static_cli_%s.json", glo_model_path, b)
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
		if times == 1 {
			last_eval_cli_filename := fmt.Sprintf("./application/users/global_Model/global_Model_Server/%s/cliEval.json", glo_model_path)
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
		}
		Eval_msg, err := serviceSetup1.UpdateContriInfoIOT(E)
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
		command := "./application/users/utils/judge_Consensus.sh "
		cmd1 := exec.Command("/bin/bash", "-c", command)
		output, err1 := cmd1.Output()
		if err1 != nil {
			fmt.Printf("Judge_Consensus at round %d times %dth--->: failed with error:%s\n", round, times, err1.Error())
			return
		}
		fmt.Printf("Judge_Consensus  at round %d times %dth--->: finished :\n%s", round, times, string(output))
		eval_cli_filename := fmt.Sprintf("./application/users/global_Model/global_Model_Server/%s/cliEval.json", glo_model_path)
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

type Pair struct {
	Key   interface{}
	Value float64
}
type PairList []Pair

func (p PairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p PairList) Len() int           { return len(p) }
func (p PairList) Less(i, j int) bool { return p[i].Value < p[j].Value }
func sortMapByValue(m map[interface{}]float64) PairList {
	p := make(PairList, len(m))
	i := 0
	for k, v := range m {
		p[i] = Pair{k, v}
		i++
	}
	sort.Sort(p)
	return p
}

func sortMapByValue2(m map[string]interface{}) PairList {
	p := make(PairList, len(m))
	i := 0
	for k, v := range m {
		p[i] = Pair{k, v.(float64)}
		i++
	}
	sort.Sort(p)
	return p
}

func getWinnerbyConsensus(consensusType string, round, cli_num int, serviceSetup1 cli_service.ServiceSetup, transactions []string, train_label bool) cli_service.Consensus_competing_result {
	var Winner cli_service.Consensus_competing_result

	if consensusType == "pow" {
		var data blockGenerate.Transactions
		var byte_trans [][]byte
		for i := 0; i < len(transactions); i++ {
			byte_trans = append(byte_trans, []byte(transactions[i]))
		}
		data.Transaction = byte_trans
		Winner = pow.PoW(cli_num, data.Transaction, train_label)

		return Winner

	} else if consensusType == "pos" {
		var data blockGenerate.Transactions

		var byte_trans [][]byte
		for i := 0; i < len(transactions); i++ {
			byte_trans = append(byte_trans, []byte(transactions[i]))
		}
		data.Transaction = byte_trans
		Winner = pos.PoS(cli_num, data.Transaction,train_label)
		return Winner
	} else if consensusType == "poAggre" {
		var data blockGenerate.Transactions

		var byte_trans [][]byte
		for i := 0; i < len(transactions); i++ {
			byte_trans = append(byte_trans, []byte(transactions[i]))
		}
		data.Transaction = byte_trans
		Winner = poAggre.PoAggre(cli_num, data.Transaction)
		return Winner
	} else if consensusType == "poShapley" {
		var data blockGenerate.Transactions

		var byte_trans [][]byte
		for i := 0; i < len(transactions); i++ {
			byte_trans = append(byte_trans, []byte(transactions[i]))
		}
		data.Transaction = byte_trans
		Winner = poShapley.PoShapley(round, cli_num, serviceSetup1, data.Transaction)
		return Winner

	} else 	if consensusType == "dpos" {
		var data blockGenerate.Transactions
		var byte_trans [][]byte
		for i := 0; i < len(transactions); i++ {
			byte_trans = append(byte_trans, []byte(transactions[i]))
		}
		data.Transaction = byte_trans
		Winner = dPoS.DPoS( data.Transaction,serviceSetup1,	train_label)

		return Winner
	} else {
		fmt.Println("the consensus type is unknow")
		return Winner
	}

}
