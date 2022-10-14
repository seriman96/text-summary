function handlePaste(e) {
    let clipboardData, pastedData;
  
    // Get pasted data via clipboard API
    clipboardData = e.clipboardData || window.clipboardData;
    pastedData = clipboardData.getData("Text");
  
    // Stop data actually being pasted into div, if it wasn't copied from within the same
    if (pastedData != e.target.copiedData) {
      e.stopPropagation();
      e.preventDefault();
    }
  }
  function handleCopy(e) {
    const textarea = e.target;
    const selectionStart = textarea.selectionStart;
    const selectionEnd = textarea.selectionEnd;
    textarea.copiedData = textarea.value.substring(selectionStart, selectionEnd);
  }
  // const textareaIds = ["edit1", "edit2"];
  const textareas = document.querySelectorAll("textarea");
  for (textarea of textareas) {
    textarea.addEventListener("paste", handlePaste);
    textarea.addEventListener("copy", handleCopy);
    textarea.addEventListener("cut", handleCopy);
  }
  const button = document.querySelector("button");
  button.addEventListener("click", e => {
    console.log("paste");
    document.execCommand('insertText', false, textareas[0].copiedData)
  });
  