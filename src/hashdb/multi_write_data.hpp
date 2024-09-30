#ifndef MULTI_WRITE_DATA_HPP
#define MULTI_WRITE_DATA_HPP

#include <string>
#include <unordered_map>
#include "definitions.hpp"
#include "zklog.hpp"
#include "multi_query.hpp"

using namespace std;

class MultiWriteData
{
public:
    // Flush data
    unordered_map<string, string> program;
    unordered_map<string, string> programIntray;
    unordered_map<string, string> nodes;
    unordered_map<string, string> nodesIntray;
    string nodesStateRoot;

    // SQL queries, including all data to store in database
    MultiQuery multiQuery;

    // Indicates if data has been already stored in database
    bool stored;

    void Reset (void)
    {
        // Reset strings
        program.clear();
        programIntray.clear();
        nodes.clear();
        nodesIntray.clear();
        nodesStateRoot.clear();
        multiQuery.reset();
        stored = false;
    }

    bool IsEmpty (void)
    {
        return (nodes.size() == 0) &&
               (nodesIntray.size() == 0) &&
               (program.size() == 0) &&
               (programIntray.size() == 0) &&
               (nodesStateRoot.size() == 0);
    }

    string GetMemoryUsage (void) {
        string content = "{ ";
        size_t nodesMemUsage = sizeof(nodes);
        nodesMemUsage += nodes.bucket_count() * sizeof(void*);
        for (const auto& pair : nodes) {
            nodesMemUsage += sizeof(pair.first) + pair.first.capacity();
            nodesMemUsage += sizeof(pair.second) + pair.second.capacity();
        }
        content += "nodes=" + to_string(double(nodesMemUsage)/1024) + "Kb, ";

        size_t nodesIntrayMemUsage = sizeof(nodesIntray);
        nodesIntrayMemUsage += nodesIntray.bucket_count() * sizeof(void*);
        for (const auto& pair : nodesIntray) {
            nodesIntrayMemUsage += sizeof(pair.first) + pair.first.capacity();
            nodesIntrayMemUsage += sizeof(pair.second) + pair.second.capacity();
        }
        content += "nodesIntray=" + to_string(double(nodesIntrayMemUsage)/1024) + "Kb, ";

        size_t programMemUsage = sizeof(program);
        programMemUsage += program.bucket_count() * sizeof(void*);
        for (const auto& pair : program) {
            programMemUsage += sizeof(pair.first) + pair.first.capacity();
            programMemUsage += sizeof(pair.second) + pair.second.capacity();
        }
        content += "program=" + to_string(double(programMemUsage)/1024) + "Kb, ";

        size_t programIntrayMemUsage = sizeof(programIntray);
        programIntrayMemUsage += programIntray.bucket_count() * sizeof(void*);
        for (const auto& pair : programIntray) {
            programIntrayMemUsage += sizeof(pair.first) + pair.first.capacity();
            programIntrayMemUsage += sizeof(pair.second) + pair.second.capacity();
        }
        content += "programIntray=" + to_string(double(programIntrayMemUsage)/1024) + "Kb, ";

        content += "multiQuery=" + to_string(double(multiQuery.size())/1024) + "Kb, ";

        auto memSize = nodesMemUsage + nodesIntrayMemUsage + programMemUsage + programIntrayMemUsage + multiQuery.size();
        content += "totalMem=" + to_string(double(memSize)/1024) + "Kb }";

        return content;
    }

    void acceptIntray (bool bSenderCalling = false)
    {
        if (programIntray.size() > 0)
        {
#ifdef LOG_DB_ACCEPT_INTRAY
            if (bSenderCalling)
            {
                zklog.info("MultiWriteData::acceptIntray() rescuing " + to_string(programIntray.size()) + " program hashes");
            }
#endif
            program.merge(programIntray);
            programIntray.clear();
        }
        if (nodesIntray.size() > 0)
        {
#ifdef LOG_DB_ACCEPT_INTRAY
            if (bSenderCalling)
            {
                zklog.info("MultiWriteData::acceptIntray() rescuing " + to_string(nodesIntray.size()) + " nodes hashes");
            }
#endif
            nodes.merge(nodesIntray);
            nodesIntray.clear();
        }
    }
};

#endif