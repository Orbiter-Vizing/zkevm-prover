# zkEVM Prover

Built to interface with Ethereum Virtual Machines (EVM), the prover provides critical services through three primary RPC clients: the Aggregator client, Executor service, and StateDB service. The Aggregator client connects to an Aggregator server and harnesses multiple zkEVM Provers simultaneously, thereby maximizing proof generation efficiency. This involves a process where the Prover component calculates a resulting state by processing EVM transaction batches and subsequently generates a proof based on the PIL polynomials definition and their constraints. The Executor service offers a mechanism to validate the integrity of proposed EVM transaction batches, ensuring they adhere to specific workload requirements. The StateDB service interfaces with a system's state (represented as a Merkle tree) and the corresponding database, thus serving as a centralized state information repository.

## Components

### Aggregator client

- It establishes a connection to an Aggregator server.
- Multiple zkEVM Provers can simultaneously connect to the Aggregator server, thereby enhancing the proof generation capability.
- Upon being invoked by the Aggregator service for batch proof generation:
  - The Prover component processes the input data (a set of EVM transactions), computes the resulting state, and creates a proof based on the PIL polynomial definitions and their constraints.
  - The Executor component integrates 14 state machines to process the input data and produce evaluations of the committed polynomials, essential for proof generation. Each state machine generates its computational evidence, and intricate calculations are passed on to the subsequent state machine.
- The Prover component then invokes the Stark component to produce a proof for the committed polynomials from the Executor's state machines.
- When tasked by the Aggregator service to produce an aggregated proof:
  - The Prover component amalgamates the results of two previously computed batch or aggregated proofs, supplied by the Aggregator, to create an aggregated proof.
- When tasked by the Aggregator service to produce a final proof:
  - The Prover component uses the outcome of a prior aggregated proof, supplied by the Aggregator, to formulate a conclusive proof that can be validated.
- The server interface for this service is delineated in the file named `aggregator.proto`.

### Executor service

- The Executor component processes the input data, which comprises a batch of EVM transactions, and computes the resulting state. Notably, no proof is produced.
- This service offers a swift method to verify whether a proposed batch of transactions is correctly constructed and if it aligns with the workload that can be proven in a single batch.
- When the Executor service invokes the Executor component, only the Main state machine is utilized. This is because the committed polynomials aren't needed, given that a proof isn't generated.
- The service's interface is outlined in the `executor.proto` file.

### StateDB service

- This service provides an interface to access the system's state (represented as a Merkle tree) and the database where this state is stored.
- Both the executor and the prover rely on it as the unified source of state. It can be utilized to retrieve specific state details, such as account balances.
- The interface for this service is described in the `statedb.proto` file.

## Compiling locally

Steps to compile `zkevm-prover` locally:
### Clone repository

```sh
git clone -b stable-v3.0.2 --recursive https://github.com/UrsaMajor-t/zkevm-prover.git
cd zkevm-prover
```

### Deployment Project

This step involves configuring the necessary dependencies, which is currently only supported on **Ubuntu**. It also includes setting up large configuration files that are approximately 75GB in size. Please note that these files will occupy an additional 115GB of space once extracted. Following these setup processes, the project will be compiled.


```sh
./tools/deploy_zkevm_prover.sh
```

- If you encounter an error during the deployment process, similar to fatal error: opening dependency file xxx: No such file or directory, you can attempt to resolve the issue by re-executing the command.

### Run zkEVM Prover
_Please ensure that you run this script from the root of the zkevm-prover repository._
```sh
./tools/run_zkevm_prover.sh monitor
```
**monitor**: It will conduct a check every minute to determine if the prover process has exited abnormally. If an abnormal exit is detected, the prover process will be restarted

### Test vectors

```sh
./build/zkProver -c testvectors/config_runFile_BatchProof.json
```