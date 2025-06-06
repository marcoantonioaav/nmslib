/**
 * Non-metric Space Library
 *
 * Main developers: Bilegsaikhan Naidan, Leonid Boytsov, Yury Malkov, Ben Frederickson, David Novak
 *
 * For the complete list of contributors and further details see:
 * https://github.com/nmslib/nmslib
 *
 * Copyright (c) 2013-2018
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */
 /*
 *
 * A Hierarchical Navigable Small World (HNSW) approach.
 *
 * The main publication is (available on arxiv: http://arxiv.org/abs/1603.09320):
 * "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" 
 * Yu. A. Malkov, D. A. Yashunin
 * This code was contributed by Yu. A. Malkov. It also was used in tests from the paper.
 *
 *
 */
#pragma once

#include "index.h"
#include "params.h"

#include <condition_variable>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <unordered_set>

#define METH_HCNSW "hcnsw"
#define METH_HCNSW_SYN "Hierarchical_CNSW"

namespace similarity {


    typedef float (*EfficientDistFunc)(const float *pVect1, const float *pVect2, size_t &qty, float * TmpRes);

    enum DistFuncType2 {
      kDistTypeUnknown2 = -1,
      kL2Sqr16Ext2 = 1,
      kL2SqrExt2 = 2,
      kNormCosine2 = 3,
      kNegativeDotProduct2 = 4,
      kL1Norm2 = 5,
      kLInfNorm2 = 6
    };

    using std::string;
    using std::vector;
    using std::thread;
    using std::mutex;
    using std::unique_lock;
    using std::condition_variable;
    using std::ref;

    template <typename dist_t> class Space;
    class VisitedListPool2;
    template <typename dist_t> class HcnswNodeDistCloser;
    template <typename dist_t> class HcnswNodeDistFarther;

    class HcnswNode {
    public:
        HcnswNode(const Object *Obj, size_t id)
        {
            data_ = Obj;
            id_ = id;
        }
        ~HcnswNode(){};
        const Object *getData() { return data_; }
        template <typename dist_t>
        void getNeighborsByHeuristic1(priority_queue<HcnswNodeDistCloser<dist_t>> &resultSet1, const int NN,
                                      const Space<dist_t> *space)
        {
            if (resultSet1.size() < NN) {
                return;
            }
            priority_queue<HcnswNodeDistFarther<dist_t>> resultSet;
            priority_queue<HcnswNodeDistFarther<dist_t>> templist;
            vector<HcnswNodeDistFarther<dist_t>> returnlist;
            while (resultSet1.size() > 0) {
                resultSet.emplace(resultSet1.top().getDistance(), resultSet1.top().getMSWNodeHier());
                resultSet1.pop();
            }
            while (resultSet.size()) {
                if (returnlist.size() >= NN)
                    break;
                HcnswNodeDistFarther<dist_t> curen = resultSet.top();
                dist_t dist_to_query = curen.getDistance();
                resultSet.pop();
                bool good = true;
                for (HcnswNodeDistFarther<dist_t> curen2 : returnlist) {
                    dist_t curdist =
                        space->IndexTimeDistance(curen2.getMSWNodeHier()->getData(), curen.getMSWNodeHier()->getData());

                    // if (curdist <= dist_to_query) {
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good)
                    returnlist.push_back(curen);
                else
                    templist.push(curen);
            }

            while (returnlist.size() < NN && templist.size() > 0) {
                returnlist.push_back(templist.top());
                templist.pop();
            }

            for (HcnswNodeDistFarther<dist_t> curen2 : returnlist) {
                resultSet1.emplace(curen2.getDistance(), curen2.getMSWNodeHier());
            }
        };

        template <typename dist_t>
        void getNeighborsByHeuristic2(priority_queue<HcnswNodeDistCloser<dist_t>> &resultSet1, const int NN,
                                      const Space<dist_t> *space, int level)
        {
            if (resultSet1.size() < NN) {
                return;
            }
            priority_queue<HcnswNodeDistFarther<dist_t>> resultSet;
            // priority_queue<HcnswNodeDistFarther<dist_t>> templist;
            vector<HcnswNodeDistFarther<dist_t>> returnlist;
            // int full = resultSet1.size();
            while (resultSet1.size() > 0) {
                resultSet.emplace(resultSet1.top().getDistance(), resultSet1.top().getMSWNodeHier());
                resultSet1.pop();
            }
            int h = 0;
            while (resultSet.size()) {
                h++;
                if (returnlist.size() >= NN)
                    break;
                HcnswNodeDistFarther<dist_t> curen = resultSet.top();
                dist_t dist_to_query = curen.getDistance();
                resultSet.pop();
                bool good = true;
                for (HcnswNodeDistFarther<dist_t> curen2 : returnlist) {
                    dist_t curdist = space->IndexTimeDistance(curen2.getMSWNodeHier()->getData(), curen.getMSWNodeHier()->getData());

                    // if (curdist <= dist_to_query) {
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good)
                    returnlist.push_back(curen);
            }

            for (HcnswNodeDistFarther<dist_t> curen2 : returnlist) {
                resultSet1.emplace(curen2.getDistance(), curen2.getMSWNodeHier());
            }
        };
        template <typename dist_t>
        void getNeighborsByHeuristic3(priority_queue<HcnswNodeDistCloser<dist_t>> &resultSet1, const int NN,
                                      const Space<dist_t> *space, int level)
        {
            unordered_set<HcnswNode *> candidates;
            for (int i = resultSet1.size() - 1; i >= 0; i--) {
                // inputCopy[i] = resultSet1.top();
                candidates.insert(resultSet1.top().getMSWNodeHier());
                for (HcnswNode *n : resultSet1.top().getMSWNodeHier()->getAllFriends(level))
                    candidates.insert(n);
                resultSet1.pop();
            }
            for (HcnswNode *n : candidates) {
                if (n != this)
                    resultSet1.emplace(space->IndexTimeDistance(n->getData(), this->getData()), n);
            }

            if (resultSet1.size() < NN) {
                return;
            }

            vector<HcnswNodeDistCloser<dist_t>> inputCopy(resultSet1.size());
            vector<HcnswNodeDistCloser<dist_t>> templist;
            vector<HcnswNodeDistCloser<dist_t>> returnlist;
            vector<HcnswNodeDistCloser<dist_t>> highPriorityList;
            for (int i = resultSet1.size() - 1; i >= 0; i--) {
                inputCopy[i] = resultSet1.top();
                resultSet1.pop();
            }
            for (int i = 0; i < inputCopy.size(); i++) {
                if (highPriorityList.size() >= NN)
                    break;
                // if (returnlist.size() >= NN)
                //    break;
                HcnswNodeDistCloser<dist_t> curen = inputCopy[i];
                dist_t dist_to_query = curen.getDistance();

                int good = 2;
                for (HcnswNodeDistCloser<dist_t> curen2 : templist) {
                    dist_t curdist =
                        space->IndexTimeDistance(curen2.getMSWNodeHier()->getData(), curen.getMSWNodeHier()->getData());
                    if (curdist < dist_to_query) {
                        if (good == 2)
                            good = 1;
                        break;
                    }
                }
                for (HcnswNodeDistCloser<dist_t> curen2 : highPriorityList) {
                    dist_t curdist =
                        space->IndexTimeDistance(curen2.getMSWNodeHier()->getData(), curen.getMSWNodeHier()->getData());

                    if (curdist < dist_to_query) {
                        good = 0;
                        break;
                    }
                }
                if (good)
                    for (HcnswNodeDistCloser<dist_t> curen2 : returnlist) {
                        dist_t curdist =
                            space->IndexTimeDistance(curen2.getMSWNodeHier()->getData(), curen.getMSWNodeHier()->getData());

                        if (curdist < dist_to_query) {
                            good = 0;
                            break;
                        }
                    }

                if (good == 2)
                    highPriorityList.push_back(curen);
                else if (good == 1)
                    // if(good)
                    returnlist.push_back(curen);
                else
                    templist.push_back(curen);
            }

            for (HcnswNodeDistCloser<dist_t> curen2 : highPriorityList) {
                if (resultSet1.size() >= NN)
                    break;
                resultSet1.emplace(curen2.getDistance(), curen2.getMSWNodeHier());
            }
            for (HcnswNodeDistCloser<dist_t> curen2 : returnlist) {
                if (resultSet1.size() >= NN)
                    break;
                resultSet1.emplace(curen2.getDistance(), curen2.getMSWNodeHier());
            }
        };

        template <typename dist_t>
        void addFriendlevel(int level, HcnswNode *element, const Space<dist_t> *space, int delaunay_type)
        {
            unique_lock<mutex> lock(accessGuard_);
            for (unsigned i = 0; i < allFriends_[level].size(); i++)
                if (allFriends_[level][i] == element) {
                    cerr << "This should not happen. For some reason the elements is "
                            "already added";
                    return;
                }
            allFriends_[level].push_back(element);
            bool shrink = false;
            if (level > 0) {
                if (allFriends_[level].size() > maxsize) {
                    shrink = true;
                } else {
                    shrink = false;
                }
            } else if (allFriends_[level].size() > maxsize0) {
                shrink = true;
            } else {
                shrink = false;
            }
            if (shrink) {
                if (delaunay_type > 0) {
                    priority_queue<HcnswNodeDistCloser<dist_t>> resultSet;
                    // for (int i = 1; i < allFriends_[level].size(); i++) {
                    for (int i = 0; i < allFriends_[level].size(); i++) {
                        resultSet.emplace(space->IndexTimeDistance(this->getData(), allFriends_[level][i]->getData()),
                                          allFriends_[level][i]);
                    }
                    if (delaunay_type == 1)
                        this->getNeighborsByHeuristic1(resultSet, resultSet.size() - 1, space);
                    else if (delaunay_type == 2)
                        this->getNeighborsByHeuristic2(resultSet, resultSet.size() - 1, space, level);
                    else if (delaunay_type == 3)
                        this->getNeighborsByHeuristic3(resultSet, resultSet.size() - 1, space, level);
                    allFriends_[level].clear();

                    while (resultSet.size()) {
                        allFriends_[level].push_back(resultSet.top().getMSWNodeHier());
                        resultSet.pop();
                    }
                } else {
                    dist_t max = space->IndexTimeDistance(this->getData(), allFriends_[level][0]->getData());
                    int maxi = 0;
                    for (int i = 1; i < allFriends_[level].size(); i++) {
                        dist_t curd = space->IndexTimeDistance(this->getData(), allFriends_[level][i]->getData());
                        if (curd > max) {
                            max = curd;
                            maxi = i;
                        }
                    }
                    allFriends_[level].erase(allFriends_[level].begin() + maxi);
                }
            }
        }

        void init(int level1, int maxFriends, int maxfriendslevel0)
        {
            level = level1;
            maxsize = maxFriends;
            maxsize0 = maxfriendslevel0;
            allFriends_.resize(level + 1);
            for (int i = 0; i <= level; i++) {
                allFriends_[i].reserve(maxsize + 1);
            }
            allFriends_[0].reserve(maxsize0 + 1);
        }

        void copyDataAndLevel0LinksToOptIndex(char *mem1, size_t offsetlevels, size_t offsetData)
        {
            char *mem = mem1;
            // Level
            *((int *)(mem)) = level;
            mem += sizeof(int);

            char *memlevels = mem1 + offsetlevels;

            char *memt = memlevels;
            *((int *)(memt)) = (int)allFriends_[0].size();
            memt += sizeof(int);
            for (size_t j = 0; j < allFriends_[0].size(); j++) {
                *((int *)(memt)) = (int)allFriends_[0][j]->getId();
                memt += sizeof(int);
            }
            mem = mem1 + offsetData;
            memcpy(mem, data_->buffer(), data_->bufferlength());

            return;
        }

        void copyHigherLevelLinksToOptIndex(char *mem1, size_t offsetlevels)
        {
            char *mem = mem1;

            *((int *)(mem)) = level;
            mem += sizeof(int);

            char *memlevels = mem1 + offsetlevels;

            for (int i = 1; i <= level; i++) {
                char *memt = memlevels;
                *((int *)(memt)) = (int)allFriends_[i].size();

                memt += sizeof(int);
                for (size_t j = 0; j < allFriends_[i].size(); j++) {
                    *((int *)(memt)) = (int)allFriends_[i][j]->getId();
                    memt += sizeof(int);
                }
                memlevels += (1 + maxsize) * sizeof(int);
            };
            return;
        }
        const Object *getData() const { return data_; }
        size_t getId() const { return id_; }
        const vector<HcnswNode *> &getAllFriends(int level) const { return allFriends_[level]; }
        mutex accessGuard_;

        size_t id_;
        vector<vector<HcnswNode *>> allFriends_;

        int maxsize0;
        int maxsize;
        int level;

    private:
        const Object *data_;
    };

    //----------------------------------
    template <typename dist_t> class HcnswNodeDistFarther {
    public:
        HcnswNodeDistFarther()
        {
            distance = 0;
            element = NULL;
        }
        HcnswNodeDistFarther(dist_t di, HcnswNode *node)
        {
            distance = di;
            element = node;
        }
        ~HcnswNodeDistFarther() {}
        dist_t getDistance() const { return distance; }
        HcnswNode *getMSWNodeHier() const { return element; }
        bool operator<(const HcnswNodeDistFarther &obj1) const { return (distance > obj1.getDistance()); }
    private:
        dist_t distance;
        HcnswNode *element;
    };

    template <typename dist_t> class EvaluatedMSWNodeInt2 {
    public:
        EvaluatedMSWNodeInt2()
        {
            distance = 0;
            element = NULL;
        }
        EvaluatedMSWNodeInt2(dist_t di, int element1)
        {
            distance = di;
            element = element1;
        }
        ~EvaluatedMSWNodeInt2() {}
        dist_t getDistance() const { return distance; }
        int getMSWNodeHier() const { return element; }
        bool operator<(const EvaluatedMSWNodeInt2 &obj1) const { return (distance < obj1.getDistance()); }
    private:
        dist_t distance;

    public:
        int element;
    };

    template <typename dist_t> class HcnswNodeDistCloser {
    public:
        HcnswNodeDistCloser()
        {
            distance = 0;
            element = NULL;
        }
        HcnswNodeDistCloser(dist_t di, HcnswNode *node)
        {
            distance = di;
            element = node;
        }
        ~HcnswNodeDistCloser() {}
        dist_t getDistance() const { return distance; }
        HcnswNode *getMSWNodeHier() const { return element; }
        bool operator<(const HcnswNodeDistCloser &obj1) const { return (distance < obj1.getDistance()); }
    private:
        dist_t distance;
        HcnswNode *element;
    };

    template <typename dist_t> class Hcnsw : public Index<dist_t> {
    public:
        virtual void SaveIndex(const string &location) override;

        virtual void LoadIndex(const string &location) override;

        Hcnsw(bool PrintProgress, const Space<dist_t> &space, const ObjectVector &data);
        void CreateIndex(const AnyParams &IndexParams) override;

        ~Hcnsw();

        const std::string StrDesc() const override;
        void Search(RangeQuery<dist_t> *query, IdType) const override;
        void Search(KNNQuery<dist_t> *query, IdType) const override;

        void SetQueryTimeParams(const AnyParams &) override;

    private:
        typedef std::vector<HcnswNode *> ElementList;
        void baseSearchAlgorithmOld(KNNQuery<dist_t> *query);
        void baseSearchAlgorithmV1Merge(KNNQuery<dist_t> *query);
        void SearchOld(KNNQuery<dist_t> *query, bool normalize);
        void SearchV1Merge(KNNQuery<dist_t> *query, bool normalize);

        void NormalizeVect(float *v, size_t qty) {
            float sum = 0;
            for (int i = 0; i < qty; i++) {
                sum += v[i] * v[i];
            }
            if (sum != 0.0) {
                sum = 1 / sqrt(sum);
                for (int i = 0; i < qty; i++) {
                  v[i] *= sum;
                }
            }
        }


        void SaveOptimizedIndex(std::ostream& output);
        void LoadOptimizedIndex(std::istream& input);

        void SaveRegularIndexBin(std::ostream& output);
        void LoadRegularIndexBin(std::istream& input);
        void SaveRegularIndexText(std::ostream& output);
        void LoadRegularIndexText(std::istream& input);


    public:
        void kSearchElementsWithAttemptsLevel(const Space<dist_t> *space, const Object *queryObj, size_t NN,
                                              std::priority_queue<HcnswNodeDistCloser<dist_t>> &resultSet, HcnswNode *ep,
                                              int level) const;

        void add(const Space<dist_t> *space, HcnswNode *newElement, int id_data);
        void addToElementListSynchronized(HcnswNode *newElement);

        void link(HcnswNode *first, HcnswNode *second, int level, const Space<dist_t> *space, int delaunay_type)
        {
            // We have to pass the Space, since we need to know what elements can be
            // deleted from the list
            first->addFriendlevel(level, second, space, delaunay_type);
            second->addFriendlevel(level, first, space, delaunay_type);
        }

        //
    private:
        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t efConstruction_;
        size_t ef_;
        size_t searchMethod_;
        size_t indexThreadQty_;
        const Space<dist_t> &space_;
        bool PrintProgress_;
        int delaunay_type_;
        double mult_;
        int maxlevel_;
        unsigned int enterpointId_;
        unsigned int totalElementsStored_;
        std::string levels_file_;
        std::vector<int> clustered_levels_;

        ObjectVector data_rearranged_;

        VisitedListPool2 *visitedlistpool;
        HcnswNode *enterpoint_;

        mutable mutex ElListGuard_;
        mutable mutex MaxLevelGuard_;
        ElementList ElList_;

        int vectorlength_ = 0;
        DistFuncType2 dist_func_type_ = kDistTypeUnknown2;
        bool iscosine_ = false;
        size_t offsetData_, offsetLevel0_;
        char *data_level0_memory_;
        char **linkLists_;
        size_t memoryPerObject_;
        EfficientDistFunc fstdistfunc_;

        enum AlgoType { kOld, kV1Merge, kHybrid };

        AlgoType searchAlgoType_;

    protected:
        DISABLE_COPY_AND_ASSIGN(Hcnsw);
    };

    typedef unsigned char vl_type;
    class VisitedList2 {
    public:
        vl_type curV;
        vl_type *mass;
        unsigned int numelements;

        VisitedList2(int numelements1)
        {
            curV = -1;
            numelements = numelements1;
            // we allocate an extra sentinel element to prevent prefetch from accessing out of range memory
            mass = new vl_type[numelements + 1];
        }
        void reset()
        {
            curV++;
            if (curV == 0) {
                memset(mass, 0, sizeof(vl_type) * numelements);
                curV++;
            }
        };
        ~VisitedList2() { delete [] mass; }
    };
    ///////////////////////////////////////////////////////////
    //
    // Class for multi-threaded pool-management of VisitedLists
    //
    /////////////////////////////////////////////////////////

    class VisitedListPool2 {
        deque<VisitedList2 *> pool;
        mutex poolguard;
        int numelements;

    public:
        VisitedListPool2(int initmaxpools, int numelements1)
        {
            numelements = numelements1;
            for (int i = 0; i < initmaxpools; i++)
                pool.push_front(new VisitedList2(numelements));
        }
        VisitedList2 *getFreeVisitedList()
        {
            VisitedList2 *rez;
            {
                unique_lock<mutex> lock(poolguard);
                if (pool.size() > 0) {
                    rez = pool.front();
                    pool.pop_front();
                } else {
                    rez = new VisitedList2(numelements);
                }
            }
            rez->reset();
            return rez;
        };
        void releaseVisitedList(VisitedList2 *vl)
        {
            unique_lock<mutex> lock(poolguard);
            pool.push_front(vl);
        };
        ~VisitedListPool2()
        {
            //LOG(LIB_INFO) << "Total " << pool.size() << " lists allocated";
            while (pool.size()) {
                VisitedList2 *rez = pool.front();
                pool.pop_front();
                delete rez;
            }
        };
    };
}
