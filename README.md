# deepseek-runner

### Create and activate virtual environment

```Bash
python -m venv .venv
source .venv/bin/activate
```

### Instal requirements
```Bash
pip install -r requirements.txt
```

### Running the model
Execute this command from the directory where you have cloned the repo.
```Bash
python runner.py
```
Optional: Install CUDA tollkit if using GPU

### Output
```Bash
  Response: Write a Python function to implement binary search.
  
  
  def binary_search(arr, x):
      low = 0
      high = len(arr) - 1
      mid = 0
  
      while low <= high:
  
          mid = (high + low) // 2
  
          if arr[mid] < x:
              low = mid + 1
  
          elif arr[mid] > x:
              high = mid - 1
  
          else:
              return mid
  
      return -1
  
  
  arr = [2, 3, 4, 10, 40]
  x = 10
  
  result = binary_search(arr, x)
  
  if result != -1:
      print("Element is present at index", str(result))
  else:
      print("Element is not present in array")
```
