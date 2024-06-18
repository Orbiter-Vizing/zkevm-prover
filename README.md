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
git clone -b stable-v6.0.0 --recursive https://github.com/Orbiter-Vizing/zkevm-prover.git
cd zkevm-prover
```

> **Please ensure that the following scripts are run from the root directory of the zkevm-prover repository.**

### Deployment Project

The setup involves configuring dependencies on **Ubuntu** and downloading large configuration files (~62GB). After extraction, these files will occupy around 86GB. Ensure at least 200GB of disk space for smooth operation. Following this, the project will be compiled.


```sh
./tools/deploy_zkevm_prover.sh
```

- If you encounter an error during the deployment process, similar to fatal error: opening dependency file xxx: No such file or directory, you can attempt to resolve the issue by re-executing the command.

### Run zkEVM Prover

```sh
./tools/run_zkevm_prover.sh monitor
```
**monitor**: It will conduct a check every minute to determine if the prover process has exited abnormally. If an abnormal exit is detected, the prover process will be restarted

### Test vectors

```sh
./build/zkProver -c config/test_batch_proof.json
```