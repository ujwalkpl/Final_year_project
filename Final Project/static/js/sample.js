fetch("sample-file.json")
  .then(Response => Response.json())
  .then(data => {
    console.log(data);
    // document.write((data.value+3));
    let main = document.querySelector('.main')
    let p = main.querySelector('p')
    p.textContent = data.value
  })