module FeatureSelection

using ParallelGA
using NearestNeighbors

export load_data

type FsEntity <: Entity
  bitdata::Array
  numNodes::Int
  fitness::Float64
  dominates::Function
  function FsEntity(bitdata::Array)
    this = new()
    this.bitdata = bitdata
    numNodes = countnz(bitdata)
    this.fitness = -1
    this.dominates = function(o_entity::FsEntity)
      if this.fitness <= o_entity.fitness && this.numNodes <= o_entity.numNodes
        if this.fitness != o_entity.fitness || this.numNodes != o_entity.numNodes
          return true
        end
        return false
      end
      return false
    end
    return this
  end
end

function create_entity(num)
  FsEntity(rand(0:1, 34))
end

function fitness(ent)
  # kNN
  k = 3
  index = find(ent.bitdata)
  error = knn_classification(train_set[index, :], test_set[index, :], k)
  println("error: $(error)")
  return error
end

function knn_classification(train, test, k)
  predict = zeros(251)
  kdtree = KDTree(train)
  for i=1:251
    idxs, dists = knn(kdtree, test[:,i], k, true)
    if sum(train_label[idxs]) > length(idxs)/2
      predict[i] = 1
    end
  end
  mean(predict .!= test_label)
end

function group_entities(pop)
  for i in pop.paretoFront
    produce(i)
  end
end

function crossover(group)
  count = length(group)

  parent1 = group[rand(1:count)]
  parent2 = group[rand(1:count)]

  FsEntity(parent1.bitdata $ parent2.bitdata)
end

function mutate(ent)
  count = length(ent.bitdata)
  ent.bitdata[rand(1:count)] = rand(0:1)
end

# function test_serial()
#   runga(1, 100, 100, 0.1, 10, .1)
# end
#
# function test_parallel(nprocs_to_add = 2)
#   addprocs(nprocs_to_add)
#   require("testPGA.jl")
#   println("nprocs: $(nprocs())")
#   runga(1, 100, 100, 0.1, 10, .1)
# end
function load_data()
  f = open("ionosphere.data")
  dataset = zeros(34, 351)
  lines = readlines(f)
  label = zeros(Int, 351)

  for (ind, ln) in enumerate(lines)
    arr = split(ln, ",")
    b = [parse(Float64, i) for i in arr[1:end-1]]
    dataset[:, ind] = b
    if arr[end] == "g\n"
      label[ind] = 1
    end
  end
  close(f)
  global train_set = dataset[:, 1:100]
  global test_set = dataset[:, 101:351]
  global train_label = label[1:100]
  global test_label = label[101:351]
end

end
